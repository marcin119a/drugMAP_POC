import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

from dataset import DrugDataset, build_drug2idx, build_drug_fingerprints
from losses import compute_loss
from model import DrugMAPVAE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Average LN_IC50 for duplicate (CELL_LINE_NAME, DRUG_NAME) pairs."""
    ic50_mean = df.groupby(["CELL_LINE_NAME", "DRUG_NAME"])["LN_IC50"].mean()
    deduped = df.drop_duplicates(subset=["CELL_LINE_NAME", "DRUG_NAME"]).copy()
    deduped = deduped.set_index(["CELL_LINE_NAME", "DRUG_NAME"])
    deduped["LN_IC50"] = ic50_mean
    return deduped.reset_index()


def train_epoch(model, loader, optimizer, device, w_kl, w_ic50, w_mmd):
    model.train()
    totals = {"recon": 0, "kl": 0, "ic50": 0, "mmd": 0, "total": 0}
    n = 0

    for x, drug_fp, ic50, domain in loader:
        x, drug_fp, ic50 = x.to(device), drug_fp.to(device), ic50.to(device)
        domain = domain.to(device)

        recon, mu, logvar, z, ic50_pred = model(x, drug_fp)

        mask_cell = domain == 0
        z_cell = z[mask_cell]
        z_patient = z[~mask_cell]

        loss, parts = compute_loss(
            recon, x, mu, logvar,
            ic50_pred, ic50, z_cell, z_patient, mask_cell,
            w_kl=w_kl, w_ic50=w_ic50, w_mmd=w_mmd,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = x.size(0)
        n += bs
        totals["total"] += loss.item() * bs
        for k, v in parts.items():
            totals[k] += v * bs

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(model, loader, device, w_kl, w_ic50, w_mmd):
    model.eval()
    totals = {"recon": 0, "kl": 0, "ic50": 0, "mmd": 0, "total": 0}
    n = 0

    for x, drug_fp, ic50, domain in loader:
        x, drug_fp, ic50 = x.to(device), drug_fp.to(device), ic50.to(device)
        domain = domain.to(device)

        recon, mu, logvar, z, ic50_pred = model(x, drug_fp)

        mask_cell = domain == 0
        z_cell = z[mask_cell]
        z_patient = z[~mask_cell]

        loss, parts = compute_loss(
            recon, x, mu, logvar,
            ic50_pred, ic50, z_cell, z_patient, mask_cell,
            w_kl=w_kl, w_ic50=w_ic50, w_mmd=w_mmd,
        )

        bs = x.size(0)
        n += bs
        totals["total"] += loss.item() * bs
        for k, v in parts.items():
            totals[k] += v * bs

    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccle", default="data/ccle_filtered_merged_df.parquet")
    parser.add_argument("--patient", default="data/filtered_merged_df.parquet")
    parser.add_argument("--smiles", default="data/drug_smiles.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--drug_emb_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w_kl", type=float, default=1e-3)
    parser.add_argument("--w_ic50", type=float, default=1.0)
    parser.add_argument("--w_mmd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--save", default="checkpoints/drugmap.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ccle_df = deduplicate(pd.read_parquet(args.ccle))
    patient_df = deduplicate(pd.read_parquet(args.patient))
    drug2idx = build_drug2idx(ccle_df, patient_df)
    drug_names = [d for d, _ in sorted(drug2idx.items(), key=lambda x: x[1])]

    rna_sample = ccle_df["TPM"].iloc[0]
    mut_sample = ccle_df["mutation"].iloc[0]
    input_dim = len(rna_sample) + len(mut_sample)
    print(f"Input dim: {input_dim}  (RNA: {len(rna_sample)}, mutations: {len(mut_sample)})")
    print(f"Drugs: {len(drug2idx)}")

    if not os.path.exists(args.smiles):
        raise FileNotFoundError(
            f"{args.smiles} not found — run: python scripts/fetch_smiles.py"
        )
    drug_fps = build_drug_fingerprints(args.smiles, drug_names)
    print(f"Drug fingerprints: {drug_fps.shape}")

    def split_by_cell_line(df, val_cells):
        mask = df["CELL_LINE_NAME"].isin(val_cells)
        return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)

    all_cell_lines = np.array(sorted(ccle_df["CELL_LINE_NAME"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_cell_lines)
    cut = int(len(all_cell_lines) * args.val_frac)
    val_cells = set(all_cell_lines[:cut])
    print(f"Cell lines: {len(all_cell_lines)} total, {cut} val, {len(all_cell_lines)-cut} train")

    ccle_train, ccle_val = split_by_cell_line(ccle_df, val_cells)
    pat_train, pat_val = split_by_cell_line(patient_df, val_cells)

    train_set = ConcatDataset([
        DrugDataset(ccle_train, drug2idx, drug_fps, domain=0),
        DrugDataset(pat_train, drug2idx, drug_fps, domain=1),
    ])
    val_set = ConcatDataset([
        DrugDataset(ccle_val, drug2idx, drug_fps, domain=0),
        DrugDataset(pat_val, drug2idx, drug_fps, domain=1),
    ])

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    model = DrugMAPVAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        drug_emb_dim=args.drug_emb_dim,
        fp_dim=drug_fps.shape[1],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, args.w_kl, args.w_ic50, args.w_mmd)
        va = eval_epoch(model, val_loader, device, args.w_kl, args.w_ic50, args.w_mmd)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train total={tr['total']:.4f} recon={tr['recon']:.4f} kl={tr['kl']:.4f} ic50={tr['ic50']:.4f} mmd={tr['mmd']:.4f} | "
            f"val total={va['total']:.4f} recon={va['recon']:.4f} kl={va['kl']:.4f} ic50={va['ic50']:.4f} mmd={va['mmd']:.4f}"
        )

        if va["total"] < best_val:
            best_val = va["total"]
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "args": vars(args)}, args.save)

    print(f"\nBest val loss: {best_val:.4f}  →  saved to {args.save}")


if __name__ == "__main__":
    main()
