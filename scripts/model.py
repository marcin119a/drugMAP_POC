import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.z_head = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor):
        return self.z_head(self.net(x))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DrugEncoder(nn.Module):
    def __init__(self, fp_dim: int, drug_emb_dim: int,
                 n_targets: int, n_pathways: int,
                 target_emb_dim: int = 32, pathway_emb_dim: int = 16):
        super().__init__()
        self.target_emb = nn.Embedding(n_targets, target_emb_dim)
        self.pathway_emb = nn.Embedding(n_pathways, pathway_emb_dim)
        in_dim = fp_dim + target_emb_dim + pathway_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, drug_emb_dim),
        )

    def forward(self, fp: torch.Tensor, target_idx: torch.Tensor, pathway_idx: torch.Tensor) -> torch.Tensor:
        t = self.target_emb(target_idx)
        p = self.pathway_emb(pathway_idx)
        return self.net(torch.cat([fp, t, p], dim=-1))


class DrugResponseHead(nn.Module):
    def __init__(self, latent_dim: int, drug_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + drug_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, drug_emb], dim=-1)).squeeze(-1)


class DrugMAPAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        drug_emb_dim: int,
        n_targets: int,
        n_pathways: int,
        fp_dim: int = 2048,
        target_emb_dim: int = 32,
        pathway_emb_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, dropout)
        self.drug_encoder = DrugEncoder(
            fp_dim, drug_emb_dim, n_targets, n_pathways, target_emb_dim, pathway_emb_dim
        )
        self.drug_head = DrugResponseHead(latent_dim, drug_emb_dim, hidden_dim // 4)

    def forward(self, x: torch.Tensor, drug_fp: torch.Tensor,
                target_idx: torch.Tensor, pathway_idx: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        drug_emb = self.drug_encoder(drug_fp, target_idx, pathway_idx)
        ic50_pred = self.drug_head(z, drug_emb)
        return recon, z, ic50_pred
