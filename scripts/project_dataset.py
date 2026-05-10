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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ProjectCellLineDataset(Dataset):
    """
    Cell-line drug response dataset from primary_site_train.parquet.
    domain=0 → IC50 loss is active.
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
        rna = np.stack(df["rna_vector"].values).astype(np.float32)
        mut = np.stack(df["mutation_vector"].values).astype(np.float32)
        self.features = torch.from_numpy(np.concatenate([rna, mut], axis=1))

        d_idx = np.array([drug2idx[d] for d in df["DRUG_NAME"]])
        self.drug_fp = torch.from_numpy(drug_fps[d_idx])

        self.target_idx = torch.tensor(
            [drug2target.get(d, 0) for d in df["DRUG_NAME"]], dtype=torch.long
        )
        self.pathway_idx = torch.tensor(
            [drug2pathway.get(d, 0) for d in df["DRUG_NAME"]], dtype=torch.long
        )
        self.ic50 = torch.tensor(df["LN_IC50"].values, dtype=torch.float32)
        self.cancer_label = torch.tensor(
            [cancer2idx.get(c, 0) for c in df["primary_site"]], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.ic50)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.drug_fp[idx],
            self.target_idx[idx],
            self.pathway_idx[idx],
            self.ic50[idx],
            0,  # domain — cell line
            self.cancer_label[idx],
        )


class ProjectTCGADataset(Dataset):
    """
    Patient dataset from primary_site_tcga.parquet.
    domain=1 → IC50 loss is masked out in training.
    Each patient is paired with `drugs_per_patient` randomly sampled drugs
    from the cell-line training set for contrastive learning.
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
        rna = np.stack(df["rna_vector"].values).astype(np.float32)
        mut = np.stack(df["mutation_vector"].values).astype(np.float32)
        patient_features = np.concatenate([rna, mut], axis=1)

        all_drugs = sorted(drug2idx.keys())
        rng = np.random.default_rng(seed)

        feat_rows, fp_rows, target_rows, pathway_rows, cancer_rows = [], [], [], [], []

        for i, (_, row) in enumerate(df.iterrows()):
            sampled = rng.choice(all_drugs, size=min(drugs_per_patient, len(all_drugs)), replace=False)
            cancer_id = cancer2idx.get(row["primary_site"], 0)
            for d in sampled:
                feat_rows.append(patient_features[i])
                fp_rows.append(drug_fps[drug2idx[d]])
                target_rows.append(drug2target.get(d, 0))
                pathway_rows.append(drug2pathway.get(d, 0))
                cancer_rows.append(cancer_id)

        self.features = torch.from_numpy(np.stack(feat_rows))
        self.drug_fp = torch.from_numpy(np.stack(fp_rows))
        self.target_idx = torch.tensor(target_rows, dtype=torch.long)
        self.pathway_idx = torch.tensor(pathway_rows, dtype=torch.long)
        self.cancer_label = torch.tensor(cancer_rows, dtype=torch.long)
        # IC50 placeholder — ignored by loss when domain=1
        self.ic50 = torch.zeros(len(feat_rows), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ic50)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.drug_fp[idx],
            self.target_idx[idx],
            self.pathway_idx[idx],
            self.ic50[idx],
            1,  # domain — patient
            self.cancer_label[idx],
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

    tcga_idx = np.arange(len(tcga_df))
    rng.shuffle(tcga_idx)
    tcga_cut = int(len(tcga_df) * val_frac)
    tcga_train = tcga_df.iloc[tcga_idx[tcga_cut:]].reset_index(drop=True)
    tcga_val = tcga_df.iloc[tcga_idx[:tcga_cut]].reset_index(drop=True)

    ds_kwargs = dict(
        drug_fps=drug_fps, drug2idx=drug2idx, cancer2idx=cancer2idx,
        drug2target=drug2target, drug2pathway=drug2pathway,
    )

    train_set = ConcatDataset([
        ProjectCellLineDataset(cl_train, **ds_kwargs),
        ProjectTCGADataset(tcga_train, **ds_kwargs, drugs_per_patient=drugs_per_patient, seed=seed),
    ])
    val_set = ConcatDataset([
        ProjectCellLineDataset(cl_val, **ds_kwargs),
        ProjectTCGADataset(tcga_val, **ds_kwargs, drugs_per_patient=drugs_per_patient, seed=seed),
    ])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    sample_rna = train_df["rna_vector"].iloc[0]
    sample_mut = train_df["mutation_vector"].iloc[0]
    input_dim = len(sample_rna) + len(sample_mut)

    meta = dict(
        input_dim=input_dim,
        fp_dim=FP_BITS,
        n_targets=len(target2idx),
        n_pathways=len(pathway2idx),
        drug2idx=drug2idx,
        cancer2idx=cancer2idx,
        target2idx=target2idx,
        pathway2idx=pathway2idx,
    )

    print(f"[data] Cell lines: {len(all_cell_lines)} ({len(all_cell_lines)-cut} train / {cut} val)")
    print(f"[data] TCGA patients: {len(tcga_df)} ({len(tcga_train)} train / {len(tcga_val)} val)")
    print(f"[data] Train samples: {len(train_set)}  Val samples: {len(val_set)}")
    print(f"[data] Drugs: {len(drug2idx)}  Cancer types: {len(cancer2idx)}")
    print(f"[data] Input dim: {input_dim}  FP dim: {FP_BITS}")

    return train_loader, val_loader, meta

