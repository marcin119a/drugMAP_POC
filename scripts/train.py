import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

from dataset import (DrugDataset, build_cancer2idx, build_drug2idx, build_drug_fingerprints,
                     build_target2idx, build_pathway2idx, build_drug2target, build_drug2pathway)
from losses import compute_loss
from model import DrugMAPAE
from project_dataset import build_project_loaders


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


def train_epoch(model, loader, optimizer, device, w_ic50, w_contrastive, w_aux, temperature):
    model.train()
    totals = {"recon": 0, "ic50": 0, "contrastive": 0, "aux": 0, "total": 0}
    n = 0

    for x, drug_fp, target_idx, pathway_idx, ic50, domain, cancer_label in loader:

        x, drug_fp, ic50 = x.to(device), drug_fp.to(device), ic50.to(device)
        target_idx, pathway_idx = target_idx.to(device), pathway_idx.to(device)
        domain = domain.to(device)
        cancer_label = cancer_label.to(device)

        recon, z, ic50_pred, aux_loss = model(x, drug_fp, target_idx, pathway_idx)

        mask_cell = domain == 0

        loss, parts = compute_loss(
            recon, x,
            ic50_pred, ic50, z, cancer_label, mask_cell, aux_loss,
            w_ic50=w_ic50, w_contrastive=w_contrastive, w_aux=w_aux, temperature=temperature,
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
def eval_epoch(model, loader, device, w_ic50, w_contrastive, w_aux, temperature):
    model.eval()
    totals = {"recon": 0, "ic50": 0, "contrastive": 0, "aux": 0, "total": 0}
    n = 0

    for x, drug_fp, target_idx, pathway_idx, ic50, domain, cancer_label in loader:
        x, drug_fp, ic50 = x.to(device), drug_fp.to(device), ic50.to(device)
        target_idx, pathway_idx = target_idx.to(device), pathway_idx.to(device)
        domain = domain.to(device)
        cancer_label = cancer_label.to(device)

        recon, z, ic50_pred, aux_loss = model(x, drug_fp, target_idx, pathway_idx)

        mask_cell = domain == 0

        loss, parts = compute_loss(
            recon, x,
            ic50_pred, ic50, z, cancer_label, mask_cell, aux_loss,
            w_ic50=w_ic50, w_contrastive=w_contrastive, w_aux=w_aux, temperature=temperature,
        )

        bs = x.size(0)
        n += bs
        totals["total"] += loss.item() * bs
        for k, v in parts.items():
            totals[k] += v * bs

    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", action="store_true",
                        help="use data_projects pipeline (primary_site_train + TCGA)")
    parser.add_argument("--data_dir", default="data_projects")
    parser.add_argument("--drugs_per_patient", type=int, default=5)
    parser.add_argument("--ccle", default="data/ccle_filtered_merged_df.parquet")
    parser.add_argument("--patient", default="data/filtered_merged_df.parquet")
    parser.add_argument("--smiles", default="data/drug_smiles.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--drug_emb_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w_ic50", type=float, default=1.0)
    parser.add_argument("--w_contrastive", type=float, default=1.0)
    parser.add_argument("--w_aux", type=float, default=0.01)
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save", default="checkpoints/drugmap.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.project:
        smiles_csv = f"{args.data_dir}/drug_smiles.csv"
        train_loader, val_loader, meta = build_project_loaders(
            data_dir=args.data_dir,
            smiles_csv=smiles_csv,
            val_frac=args.val_frac,
            drugs_per_patient=args.drugs_per_patient,
            batch_size=args.batch_size,
            num_workers=2,
            seed=args.seed,
        )
        rna_dim = meta["rna_dim"]
        mut_dim = meta["mut_dim"]
        n_targets = meta["n_targets"]
        n_pathways = meta["n_pathways"]
        fp_dim = meta["fp_dim"]
    else:
        ccle_df = deduplicate(pd.read_parquet(args.ccle))
        patient_df = deduplicate(pd.read_parquet(args.patient))
        drug2idx = build_drug2idx(ccle_df, patient_df)
        cancer2idx = build_cancer2idx(ccle_df, patient_df)
        target2idx = build_target2idx(ccle_df)
        pathway2idx = build_pathway2idx(ccle_df)
        drug2target = build_drug2target(ccle_df, target2idx)
        drug2pathway = build_drug2pathway(ccle_df, pathway2idx)
        print(f"Cancer types: {len(cancer2idx)}  {list(cancer2idx)}")
        print(f"Targets: {len(target2idx)}  Pathways: {len(pathway2idx)}")
        drug_names = [d for d, _ in sorted(drug2idx.items(), key=lambda x: x[1])]

        rna_sample = ccle_df["TPM"].iloc[0]
        mut_sample = ccle_df["mutation"].iloc[0]
        rna_dim = len(rna_sample)
        mut_dim = len(mut_sample)
        print(f"Input dim: {rna_dim + mut_dim}  (RNA: {rna_dim}, mutations: {mut_dim})")
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

        ds_kwargs = dict(drug2idx=drug2idx, drug_fps=drug_fps,
                         drug2target=drug2target, drug2pathway=drug2pathway)
        train_set = ConcatDataset([
            DrugDataset(ccle_train, **ds_kwargs, domain=0, cancer2idx=cancer2idx),
            DrugDataset(pat_train, **ds_kwargs, domain=1, cancer2idx=cancer2idx),
        ])
        val_set = ConcatDataset([
            DrugDataset(ccle_val, **ds_kwargs, domain=0, cancer2idx=cancer2idx),
            DrugDataset(pat_val, **ds_kwargs, domain=1, cancer2idx=cancer2idx),
        ])

        pin = device.type == "cuda"
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

        n_targets = len(target2idx)
        n_pathways = len(pathway2idx)
        fp_dim = drug_fps.shape[1]

    model = DrugMAPAE(
        rna_dim=rna_dim,
        mut_dim=mut_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        drug_emb_dim=args.drug_emb_dim,
        n_targets=n_targets,
        n_pathways=n_pathways,
        fp_dim=fp_dim,
        n_experts=args.n_experts,
        top_k=args.top_k,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    best_val = float("inf")
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, args.w_ic50, args.w_contrastive, args.w_aux, args.temperature)
        va = eval_epoch(model, val_loader, device, args.w_ic50, args.w_contrastive, args.w_aux, args.temperature)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train total={tr['total']:.4f} recon={tr['recon']:.4f} ic50={tr['ic50']:.4f} "
            f"contrastive={tr['contrastive']:.4f} aux={tr['aux']:.4f} | "
            f"val total={va['total']:.4f} recon={va['recon']:.4f} ic50={va['ic50']:.4f} "
            f"contrastive={va['contrastive']:.4f} aux={va['aux']:.4f}"
        )

        if va["total"] < best_val:
            best_val = va["total"]
            patience_counter = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "args": vars(args)}, args.save)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\nBest val loss: {best_val:.4f}  →  saved to {args.save}")


if __name__ == "__main__":
    main()
