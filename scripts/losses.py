import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    SupCon loss: pulls same-cancer embeddings together across cell lines and patients.
    z: (N, D) — latent vectors
    labels: (N,) — integer cancer type indices
    """
    N = z.size(0)
    if N < 2:
        return torch.tensor(0.0, device=z.device)

    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature          # (N, N)

    # mask[i,j]=1 if same cancer type and i≠j
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    eye = torch.eye(N, device=z.device)
    pos_mask = pos_mask * (1 - eye)

    # at least one positive required
    has_pos = pos_mask.sum(1) > 0

    # log-softmax over all non-self pairs
    sim_exp = torch.exp(sim) * (1 - eye)
    log_prob = sim - torch.log(sim_exp.sum(1, keepdim=True) + 1e-8)

    # mean over positives, then mean over anchors that have a positive
    loss_per_anchor = -(pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
    return loss_per_anchor[has_pos].mean()


def compute_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    ic50_pred: torch.Tensor,
    ic50_true: torch.Tensor,
    z: torch.Tensor,
    cancer_labels: torch.Tensor,
    mask_cell: torch.Tensor,
    aux_loss: torch.Tensor,
    w_ic50: float = 1.0,
    w_contrastive: float = 1.0,
    w_aux: float = 0.01,
    temperature: float = 0.07,
):
    loss_recon = F.mse_loss(recon, x)

    loss_ic50 = torch.tensor(0.0, device=x.device)
    if mask_cell.any():
        loss_ic50 = F.mse_loss(ic50_pred[mask_cell], ic50_true[mask_cell])

    loss_contrastive = supervised_contrastive_loss(z, cancer_labels, temperature)

    total = loss_recon + w_ic50 * loss_ic50 + w_contrastive * loss_contrastive + w_aux * aux_loss
    return total, {
        "recon": loss_recon.item(),
        "ic50": loss_ic50.item(),
        "contrastive": loss_contrastive.item(),
        "aux": aux_loss.item(),
    }
