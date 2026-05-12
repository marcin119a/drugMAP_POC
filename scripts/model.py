import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """Sparse top-k Mixture of Experts with Switch-Transformer load-balancing aux loss."""

    def __init__(self, input_dim: int, output_dim: int,
                 n_experts: int = 4, top_k: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(input_dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        gate_probs = F.softmax(self.gate(x), dim=-1)           # (B, n_experts)
        topk_vals, topk_idx = gate_probs.topk(self.top_k, dim=-1)  # (B, top_k)
        topk_weights = topk_vals / topk_vals.sum(-1, keepdim=True)  # renorm to 1

        # (B, n_experts, output_dim) → gather top-k → weighted sum
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, expert_outs.size(-1))
        out = (expert_outs.gather(1, idx_exp) * topk_weights.unsqueeze(-1)).sum(1)

        # Load-balancing aux loss (Switch Transformer): n_experts * sum(f_i * p_i)
        dispatch = torch.zeros(B, self.n_experts, device=x.device).scatter_add(
            1, topk_idx, torch.ones_like(topk_idx, dtype=torch.float)
        )
        f = dispatch.mean(0) / self.top_k   # fraction of tokens per expert, sums to 1
        p = gate_probs.mean(0)              # mean gate probability per expert
        aux_loss = self.n_experts * (f.detach() * p).sum()

        return out, aux_loss


class ModalityEncoder(nn.Module):
    """Single-modality encoder: Linear projection → MoE hidden layer → latent head."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 n_experts: int = 4, top_k: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.moe = MoELayer(hidden_dim, hidden_dim // 2, n_experts, top_k, dropout)
        self.z_head = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, aux = self.moe(self.proj(x))
        return self.z_head(h), aux


class ModalityDecoder(nn.Module):
    """Single-modality decoder: latent → MoE hidden layer → reconstruction head."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 n_experts: int = 4, top_k: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.moe = MoELayer(hidden_dim // 2, hidden_dim, n_experts, top_k, dropout)
        self.out_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, aux = self.moe(self.proj(z))
        return self.out_head(h), aux


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

    def forward(self, fp: torch.Tensor, target_idx: torch.Tensor,
                pathway_idx: torch.Tensor) -> torch.Tensor:
        t = self.target_emb(target_idx)
        p = self.pathway_emb(pathway_idx)
        return self.net(torch.cat([fp, t, p], dim=-1))


class DrugResponseHead(nn.Module):
    def __init__(self, latent_dim: int, drug_emb_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + drug_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, drug_emb], dim=-1)).squeeze(-1)


class DrugMAPAE(nn.Module):
    """
    Dual-modality autoencoder with per-modality Mixture-of-Experts.

    Two separate MoE encoders (RNA, mutations) produce independent latents that
    are fused into a shared z. Two separate MoE decoders reconstruct each
    modality from z. IC50 prediction uses z + drug embedding.
    """

    def __init__(
        self,
        rna_dim: int,
        mut_dim: int,
        hidden_dim: int,
        latent_dim: int,
        drug_emb_dim: int,
        n_targets: int,
        n_pathways: int,
        fp_dim: int = 2048,
        target_emb_dim: int = 32,
        pathway_emb_dim: int = 16,
        n_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rna_dim = rna_dim
        self.mut_dim = mut_dim

        moe_kw = dict(n_experts=n_experts, top_k=top_k, dropout=dropout)

        self.rna_encoder = ModalityEncoder(rna_dim, hidden_dim, latent_dim, **moe_kw)
        self.mut_encoder = ModalityEncoder(mut_dim, hidden_dim, latent_dim, **moe_kw)
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)

        self.rna_decoder = ModalityDecoder(latent_dim, hidden_dim, rna_dim, **moe_kw)
        self.mut_decoder = ModalityDecoder(latent_dim, hidden_dim, mut_dim, **moe_kw)

        self.drug_encoder = DrugEncoder(
            fp_dim, drug_emb_dim, n_targets, n_pathways, target_emb_dim, pathway_emb_dim
        )
        self.drug_head = DrugResponseHead(latent_dim, drug_emb_dim, hidden_dim // 4, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        drug_fp: torch.Tensor,
        target_idx: torch.Tensor,
        pathway_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rna = x[:, : self.rna_dim]
        mut = x[:, self.rna_dim :]

        z_rna, aux_enc_rna = self.rna_encoder(rna)
        z_mut, aux_enc_mut = self.mut_encoder(mut)
        z = self.fusion(torch.cat([z_rna, z_mut], dim=-1))

        recon_rna, aux_dec_rna = self.rna_decoder(z)
        recon_mut, aux_dec_mut = self.mut_decoder(z)
        recon = torch.cat([recon_rna, recon_mut], dim=-1)

        drug_emb = self.drug_encoder(drug_fp, target_idx, pathway_idx)
        ic50_pred = self.drug_head(z, drug_emb)

        aux_loss = aux_enc_rna + aux_enc_mut + aux_dec_rna + aux_dec_mut
        return recon, z, ic50_pred, aux_loss
