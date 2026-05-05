import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DrugEncoder(nn.Module):
    def __init__(self, fp_dim: int, drug_emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, fp_dim // 4),
            nn.ReLU(),
            nn.Linear(fp_dim // 4, drug_emb_dim),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net(fp)


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


class DrugMAPVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        drug_emb_dim: int,
        fp_dim: int = 2048,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.drug_encoder = DrugEncoder(fp_dim, drug_emb_dim)
        self.drug_head = DrugResponseHead(latent_dim, drug_emb_dim, hidden_dim // 4)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, drug_fp: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        drug_emb = self.drug_encoder(drug_fp)
        ic50_pred = self.drug_head(z, drug_emb)
        return recon, mu, logvar, z, ic50_pred
