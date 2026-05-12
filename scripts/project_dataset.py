from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from dataset import FP_BITS, build_drug_fingerprints


# ---------------------------------------------------------------------------
# Lookup dict builders (from primary_site_train)
# ---------------------------------------------------------------------------

def build_dicts(train_df: pd.DataFrame) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Returns drug2idx, cancer2idx, target2idx, pathway2idx, drug2target, drug2pathway."""
    drug2idx = {d: i for i, d in enumerate(sorted(train_df["DRUG_NAME"].unique()))}
    cancer2idx = {c: i for i, c in enumerate(sorted(train_df["primary_site"].unique()))}

    targets = sorted(train_df["PUTATIVE_TARGET"].dropna().unique())
    target2idx = {"UNK": 0, **{t: i + 1 for i, t in enumerate(targets)}}

    pathways = sorted(train_df["PATHWAY_NAME"].dropna().unique())
    pathway2idx = {"UNK": 0, **{p: i + 1 for i, p in enumerate(pathways)}}

    drug_meta = train_df[["DRUG_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME"]].drop_duplicates("DRUG_NAME")
    drug2target = {r["DRUG_NAME"]: target2idx.get(r["PUTATIVE_TARGET"], 0) for _, r in drug_meta.iterrows()}
    drug2pathway = {r["DRUG_NAME"]: pathway2idx.get(r["PATHWAY_NAME"], 0) for _, r in drug_meta.iterrows()}

    return drug2idx, cancer2idx, target2idx, pathway2idx, drug2target, drug2pathway


def _extract_features(df: pd.DataFrame) -> np.ndarray:
    """Stack rna+mutation vectors into a contiguous float32 matrix and free intermediates."""
    rna = np.nan_to_num(np.stack(df["rna_vector"].values).astype(np.float32), nan=0.0)
    mut = np.stack(df["mutation_vector"].values).astype(np.float32)
    out = np.concatenate([rna, mut], axis=1)
    del rna, mut
    return out


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ProjectCellLineDataset(Dataset):
    """
    Cell-line drug response dataset from primary_site_train.parquet.
    domain=0 → IC50 loss is active.

    Stores features as numpy arrays; torch tensors are created per-sample in
    __getitem__ via zero-copy from_numpy, avoiding a doubled in-memory footprint.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        drug_fps: np.ndarray,
        drug2idx: dict,
        cancer2idx: dict,
        drug2target: dict,
        drug2pathway: dict,
    ):
        self.features = _extract_features(df)

        d_idx = np.array([drug2idx[d] for d in df["DRUG_NAME"]])
        self.drug_fp = drug_fps[d_idx]  # view into shared fp matrix

        self.target_idx = np.array([drug2target.get(d, 0) for d in df["DRUG_NAME"]], dtype=np.int64)
        self.pathway_idx = np.array([drug2pathway.get(d, 0) for d in df["DRUG_NAME"]], dtype=np.int64)
        self.ic50 = torch.from_numpy(df["LN_IC50"].values.astype(np.float32))
        self.cancer_label = np.array([cancer2idx.get(c, 0) for c in df["primary_site"]], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ic50)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx].copy()),
            torch.from_numpy(self.drug_fp[idx].copy()),
            int(self.target_idx[idx]),
            int(self.pathway_idx[idx]),
            self.ic50[idx],
            0,  # domain — cell line
            int(self.cancer_label[idx]),
        )


class ProjectTCGADataset(Dataset):
    """
    Patient dataset from primary_site_tcga.parquet.
    domain=1 → IC50 loss is masked out in training.

    Patient features are stored once (N_patients × dim). Drug pairings are stored
    as index arrays, so each feature vector is NOT duplicated drugs_per_patient times.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        drug_fps: np.ndarray,
        drug2idx: dict,
        cancer2idx: dict,
        drug2target: dict,
        drug2pathway: dict,
        drugs_per_patient: int = 5,
        seed: int = 42,
    ):
        self.patient_features = _extract_features(df)  # (N_patients, dim) — stored once
        self.drug_fps = drug_fps  # shared reference, not copied

        all_drugs = sorted(drug2idx.keys())
        n_drugs = len(all_drugs)
        k = min(drugs_per_patient, n_drugs)
        n_patients = len(df)

        rng = np.random.default_rng(seed)
        # Shape: (n_patients, k) — indices into all_drugs list
        sampled_slots = np.array([
            rng.choice(n_drugs, size=k, replace=False)
            for _ in range(n_patients)
        ])

        all_drug_names = np.array(all_drugs)

        # Flat index arrays — one entry per (patient, drug) pair
        self.patient_idx = np.repeat(np.arange(n_patients), k)
        sampled_drug_names = all_drug_names[sampled_slots.flatten()]

        self.drug_fp_idx = np.array([drug2idx[d] for d in sampled_drug_names], dtype=np.int64)
        self.target_idx = np.array([drug2target.get(d, 0) for d in sampled_drug_names], dtype=np.int64)
        self.pathway_idx = np.array([drug2pathway.get(d, 0) for d in sampled_drug_names], dtype=np.int64)

        cancer_per_patient = np.array(
            [cancer2idx.get(row["primary_site"], 0) for _, row in df.iterrows()], dtype=np.int64
        )
        self.cancer_label = cancer_per_patient[self.patient_idx]

    def __len__(self) -> int:
        return len(self.patient_idx)

    def __getitem__(self, idx):
        p = self.patient_idx[idx]
        d = self.drug_fp_idx[idx]
        return (
            torch.from_numpy(self.patient_features[p].copy()),
            torch.from_numpy(self.drug_fps[d].copy()),
            int(self.target_idx[idx]),
            int(self.pathway_idx[idx]),
            torch.tensor(0., dtype=torch.float32),  # IC50 placeholder — ignored by loss when domain=1
            1,    # domain — patient
            int(self.cancer_label[idx]),
        )


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

def build_project_loaders(
    data_dir: str = "data_projects",
    smiles_csv: str = "data_projects/drug_smiles.csv",
    val_frac: float = 0.2,
    drugs_per_patient: int = 5,
    batch_size: int = 512,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Load data from `data_dir` and return (train_loader, val_loader, meta).

    Requires `smiles_csv` (DRUG_NAME, SMILES columns) — generate with:
        python scripts/fetch_smiles.py --train data_projects/primary_site_train.parquet \\
                                       --out data_projects/drug_smiles.csv

    meta keys: input_dim, n_targets, n_pathways, fp_dim,
               drug2idx, cancer2idx, target2idx, pathway2idx
    """
    data_dir = Path(data_dir)

    train_df = pd.read_parquet(data_dir / "primary_site_train.parquet")
    tcga_df = pd.read_parquet(data_dir / "primary_site_tcga.parquet")

    drug2idx, cancer2idx, target2idx, pathway2idx, drug2target, drug2pathway = build_dicts(train_df)
    drug_names = sorted(drug2idx.keys())

    if not Path(smiles_csv).exists():
        raise FileNotFoundError(
            f"{smiles_csv} not found — run:\n"
            f"  python scripts/fetch_smiles.py "
            f"--train data_projects/primary_site_train.parquet "
            f"--out {smiles_csv}"
        )
    drug_fps = build_drug_fingerprints(smiles_csv, drug_names)

    # --- train/val split by cell line ---
    all_cell_lines = np.array(sorted(train_df["CELL_LINE_NAME"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_cell_lines)
    cut = int(len(all_cell_lines) * val_frac)
    val_cells = set(all_cell_lines[:cut])

    cl_train = train_df[~train_df["CELL_LINE_NAME"].isin(val_cells)].reset_index(drop=True)
    cl_val = train_df[train_df["CELL_LINE_NAME"].isin(val_cells)].reset_index(drop=True)

    # Measure dims before dropping columns
    sample_rna = train_df["rna_vector"].iloc[0]
    sample_mut = train_df["mutation_vector"].iloc[0]
    rna_dim = len(sample_rna)
    mut_dim = len(sample_mut)
    input_dim = rna_dim + mut_dim

    # Free the large object-dtype columns from both full dataframes — the subsets
    # (cl_train / cl_val) will still carry them until the Datasets are constructed.
    del train_df, tcga_df

    tcga_df = pd.read_parquet(data_dir / "primary_site_tcga.parquet")
    tcga_idx = np.arange(len(tcga_df))
    rng.shuffle(tcga_idx)
    tcga_cut = int(len(tcga_df) * val_frac)
    tcga_train = tcga_df.iloc[tcga_idx[tcga_cut:]].reset_index(drop=True)
    tcga_val = tcga_df.iloc[tcga_idx[:tcga_cut]].reset_index(drop=True)
    del tcga_df

    ds_kwargs = dict(
        drug_fps=drug_fps, drug2idx=drug2idx, cancer2idx=cancer2idx,
        drug2target=drug2target, drug2pathway=drug2pathway,
    )

    train_set = ConcatDataset([
        ProjectCellLineDataset(cl_train, **ds_kwargs),
        ProjectTCGADataset(tcga_train, **ds_kwargs, drugs_per_patient=drugs_per_patient, seed=seed),
    ])
    del cl_train, tcga_train

    val_set = ConcatDataset([
        ProjectCellLineDataset(cl_val, **ds_kwargs),
        ProjectTCGADataset(tcga_val, **ds_kwargs, drugs_per_patient=drugs_per_patient, seed=seed),
    ])
    del cl_val, tcga_val

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    n_cell_lines = len(all_cell_lines)
    print(f"[data] Cell lines: {n_cell_lines} ({n_cell_lines - cut} train / {cut} val)")
    print(f"[data] Train samples: {len(train_set)}  Val samples: {len(val_set)}")
    print(f"[data] Drugs: {len(drug2idx)}  Cancer types: {len(cancer2idx)}")
    print(f"[data] Input dim: {input_dim}  FP dim: {FP_BITS}")

    meta = dict(
        input_dim=input_dim,
        rna_dim=rna_dim,
        mut_dim=mut_dim,
        fp_dim=FP_BITS,
        n_targets=len(target2idx),
        n_pathways=len(pathway2idx),
        drug2idx=drug2idx,
        cancer2idx=cancer2idx,
        target2idx=target2idx,
        pathway2idx=pathway2idx,
    )

    return train_loader, val_loader, meta
