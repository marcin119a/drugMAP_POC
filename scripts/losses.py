import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def mmd(x: torch.Tensor, y: torch.Tensor, kernel_bandwidths=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """RBF kernel MMD between two sample sets."""
    def rbf(a, b, bw):
        diff = a.unsqueeze(1) - b.unsqueeze(0)        # (n, m, d)
        return torch.exp(-diff.pow(2).sum(-1) / (2 * bw ** 2))

    k_xx = sum(rbf(x, x, bw) for bw in kernel_bandwidths) / len(kernel_bandwidths)
    k_yy = sum(rbf(y, y, bw) for bw in kernel_bandwidths) / len(kernel_bandwidths)
    k_xy = sum(rbf(x, y, bw) for bw in kernel_bandwidths) / len(kernel_bandwidths)

    n, m = x.size(0), y.size(0)
    # unbiased estimator: exclude diagonal for within-domain terms
    k_xx = (k_xx.sum() - k_xx.trace()) / (n * (n - 1))
    k_yy = (k_yy.sum() - k_yy.trace()) / (m * (m - 1))
    k_xy = k_xy.mean()

    return k_xx + k_yy - 2 * k_xy


def compute_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    ic50_pred: torch.Tensor,
    ic50_true: torch.Tensor,
    z_cell: torch.Tensor,
    z_patient: torch.Tensor,
    mask_cell: torch.Tensor,
    w_kl: float = 1e-3,
    w_ic50: float = 1.0,
    w_mmd: float = 1.0,
):
    loss_recon = F.mse_loss(recon, x)
    loss_kl = kl_divergence(mu, logvar)

    loss_ic50 = torch.tensor(0.0, device=x.device)
    if mask_cell.any():
        loss_ic50 = F.mse_loss(ic50_pred[mask_cell], ic50_true[mask_cell])

    loss_mmd = torch.tensor(0.0, device=x.device)
    if z_cell.size(0) > 1 and z_patient.size(0) > 1:
        loss_mmd = mmd(z_cell, z_patient)

    total = loss_recon + w_kl * loss_kl + w_ic50 * loss_ic50 + w_mmd * loss_mmd
    return total, {
        "recon": loss_recon.item(),
        "kl": loss_kl.item(),
        "ic50": loss_ic50.item(),
        "mmd": loss_mmd.item(),
    }
