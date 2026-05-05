import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DrugDataset(Dataset):
    """
    domain=0 → cell lines (CCLE), ic50 used in loss
    domain=1 → patients, ic50 ignored in loss
    """

    def __init__(self, df: pd.DataFrame, drug2idx: dict, domain: int):
        self.domain = domain
        rna_col = "TPM"
        mut_col = "mutation" if "mutation" in df.columns else "symbol_counts"

        rna = np.stack(df[rna_col].values).astype(np.float32)
        mut = np.stack(df[mut_col].values).astype(np.float32)
        self.features = torch.from_numpy(np.concatenate([rna, mut], axis=1))
        self.drug_idx = torch.tensor(
            [drug2idx[d] for d in df["DRUG_NAME"]], dtype=torch.long
        )
        self.ic50 = torch.tensor(df["LN_IC50"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ic50)

    def __getitem__(self, idx):
        return self.features[idx], self.drug_idx[idx], self.ic50[idx], self.domain


def build_drug2idx(ccle_df: pd.DataFrame, patient_df: pd.DataFrame) -> dict:
    drugs = sorted(set(ccle_df["DRUG_NAME"]) | set(patient_df["DRUG_NAME"]))
    return {d: i for i, d in enumerate(drugs)}
