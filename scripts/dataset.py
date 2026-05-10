import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem


FP_BITS = 2048
FP_RADIUS = 2


def smiles_to_fp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(FP_BITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)
    return np.array(fp, dtype=np.float32)


def build_drug_fingerprints(smiles_csv: str, drug_names: list[str]) -> np.ndarray:
    """Returns (n_drugs, FP_BITS) array; zero vector for drugs without SMILES."""
    df = pd.read_csv(smiles_csv).set_index("DRUG_NAME")
    fps = []
    missing = []
    for name in drug_names:
        if name in df.index and pd.notna(df.loc[name, "SMILES"]):
            fps.append(smiles_to_fp(df.loc[name, "SMILES"]))
        else:
            fps.append(np.zeros(FP_BITS, dtype=np.float32))
            missing.append(name)
    if missing:
        print(f"[dataset] No SMILES for {len(missing)} drugs, using zero vectors: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return np.stack(fps)


def build_drug2idx(ccle_df: pd.DataFrame, patient_df: pd.DataFrame) -> dict:
    drugs = sorted(set(ccle_df["DRUG_NAME"]) | set(patient_df["DRUG_NAME"]))
    return {d: i for i, d in enumerate(drugs)}


def build_cancer2idx(ccle_df: pd.DataFrame, patient_df: pd.DataFrame) -> dict:
    cancers = sorted(set(ccle_df["primary_disease"]) | set(patient_df["primary_disease"]))
    return {c: i for i, c in enumerate(cancers)}


def build_target2idx(ccle_df: pd.DataFrame) -> dict:
    targets = sorted(ccle_df["PUTATIVE_TARGET"].dropna().unique())
    return {"UNK": 0, **{t: i + 1 for i, t in enumerate(targets)}}


def build_pathway2idx(ccle_df: pd.DataFrame) -> dict:
    pathways = sorted(ccle_df["PATHWAY_NAME"].dropna().unique())
    return {"UNK": 0, **{p: i + 1 for i, p in enumerate(pathways)}}


def build_drug2target(ccle_df: pd.DataFrame, target2idx: dict) -> dict:
    rows = ccle_df[["DRUG_NAME", "PUTATIVE_TARGET"]].drop_duplicates("DRUG_NAME")
    return {r["DRUG_NAME"]: target2idx.get(r["PUTATIVE_TARGET"], 0) for _, r in rows.iterrows()}


def build_drug2pathway(ccle_df: pd.DataFrame, pathway2idx: dict) -> dict:
    rows = ccle_df[["DRUG_NAME", "PATHWAY_NAME"]].drop_duplicates("DRUG_NAME")
    return {r["DRUG_NAME"]: pathway2idx.get(r["PATHWAY_NAME"], 0) for _, r in rows.iterrows()}


class DrugDataset(Dataset):
    """
    domain=0 → cell lines (CCLE), ic50 used in loss
    domain=1 → patients, ic50 ignored in loss
    cancer_label → primary_disease integer index, used for contrastive loss
    """

    def __init__(self, df: pd.DataFrame, drug2idx: dict, drug_fps: np.ndarray,
                 domain: int, cancer2idx: dict,
                 drug2target: dict, drug2pathway: dict):
        self.domain = domain
        rna_col = "TPM"
        mut_col = "mutation" if "mutation" in df.columns else "symbol_counts"

        rna = np.stack(df[rna_col].values).astype(np.float32)
        mut = np.stack(df[mut_col].values).astype(np.float32)
        self.features = torch.from_numpy(np.concatenate([rna, mut], axis=1))

        idx = np.array([drug2idx[d] for d in df["DRUG_NAME"]])
        self.drug_fp = torch.from_numpy(drug_fps[idx])

        self.target_idx = torch.tensor(
            [drug2target.get(d, 0) for d in df["DRUG_NAME"]], dtype=torch.long
        )
        self.pathway_idx = torch.tensor(
            [drug2pathway.get(d, 0) for d in df["DRUG_NAME"]], dtype=torch.long
        )

        self.ic50 = torch.tensor(df["LN_IC50"].values, dtype=torch.float32)
        self.cancer_label = torch.tensor(
            [cancer2idx[c] for c in df["primary_disease"]], dtype=torch.long
        )

    def __len__(self):
        return len(self.ic50)

    def __getitem__(self, idx):
        return (self.features[idx], self.drug_fp[idx],
                self.target_idx[idx], self.pathway_idx[idx],
                self.ic50[idx], self.domain, self.cancer_label[idx])
