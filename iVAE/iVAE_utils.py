from bisect import bisect_right
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layer=1, hidden_dim=1024) -> None:
        super(MLP, self).__init__()
        model = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layer):
            model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Starts with warm-up, then decays LR by gamma at milestone steps."""

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_steps=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(
                    warmup_method
                )
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_steps = warmup_steps
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_steps:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_steps)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class SubjectPriorNet(nn.Module):
    """Predict p(z_s | u) params from a 5D subject signature."""

    def __init__(self, u_dim: int, z_s_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(u_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_s_dim),
        )

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_lv = self.net(u)
        return torch.chunk(mu_lv, 2, dim=-1)


class ImagePriorNet(nn.Module):
    """Predict p(z_i | img_feat) params from an image embedding."""

    def __init__(
        self,
        img_dim: int,
        z_i_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=img_dim,
            output_dim=2 * z_i_dim,
            n_layer=n_layers,
            hidden_dim=hidden_dim,
        )

    def forward(self, img_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_lv = self.net(img_feat)
        return torch.chunk(mu_lv, 2, dim=-1)


class EEGSubjectCondVAE(nn.Module):
    """Two-block VAE: image/content z_i and subject/style z_s."""

    def __init__(
        self,
        feature_dim: int,
        z_i_dim: int,
        z_s_dim: int,
        u_dim: int = 5,
        hidden_dim: int = 512,
        n_layers: int = 1,
        posterior_cond_on_subject: bool = False,
        image_prior_cond_on_image: bool = False,
        img_dim: Optional[int] = None,
        image_prior_hidden_dim: int = 128,
        image_prior_n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.z_i_dim = z_i_dim
        self.z_s_dim = z_s_dim
        self.z_total_dim = z_i_dim + z_s_dim
        self.posterior_cond_on_subject = posterior_cond_on_subject
        self.image_prior_cond_on_image = image_prior_cond_on_image

        self.enc = MLP(feature_dim, hidden_dim, n_layer=n_layers, hidden_dim=hidden_dim)
        self.enc_mu_i = nn.Linear(hidden_dim, z_i_dim)
        self.enc_lv_i = nn.Linear(hidden_dim, z_i_dim)
        s_in_dim = hidden_dim + u_dim if posterior_cond_on_subject else hidden_dim
        self.enc_mu_s = nn.Linear(s_in_dim, z_s_dim)
        self.enc_lv_s = nn.Linear(s_in_dim, z_s_dim)

        self.prior_net = SubjectPriorNet(
            u_dim=u_dim,
            z_s_dim=z_s_dim,
            hidden_dim=max(64, hidden_dim // 4),
        )
        if self.image_prior_cond_on_image:
            if img_dim is None:
                raise ValueError("img_dim must be set when image_prior_cond_on_image=True")
            self.img_prior_net = ImagePriorNet(
                img_dim=img_dim,
                z_i_dim=z_i_dim,
                hidden_dim=image_prior_hidden_dim,
                n_layers=image_prior_n_layers,
            )
        else:
            self.img_prior_net = None

        self.decoder = nn.Sequential(
            nn.Linear(self.z_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    @staticmethod
    def _sample(mu: torch.Tensor, log_var: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    @staticmethod
    def capacity_schedule(global_step: int, C_max: float, C_stop_iter: int) -> float:
        """Linear capacity ramp from 0 to C_max over C_stop_iter steps."""
        if C_stop_iter <= 0:
            return C_max
        return min(C_max * global_step / C_stop_iter, C_max)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        img_feat: Optional[torch.Tensor] = None,
        global_step: int = 0,
        C_max: float = 25.0,
        C_stop_iter: int = 10000,
    ) -> Dict[str, Any]:
        h = self.enc(x)
        q_i_mu = self.enc_mu_i(h)
        q_i_lv = self.enc_lv_i(h)
        h_s = torch.cat([h, u], dim=-1) if self.posterior_cond_on_subject else h
        q_s_mu = self.enc_mu_s(h_s)
        q_s_lv = self.enc_lv_s(h_s)

        p_s_mu, p_s_lv = self.prior_net(u)
        prior_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {"s": (p_s_mu, p_s_lv)}
        if self.image_prior_cond_on_image:
            if img_feat is None:
                raise ValueError("img_feat must be provided when image_prior_cond_on_image=True")
            p_i_mu, p_i_lv = self.img_prior_net(img_feat)
            prior_params["i"] = (p_i_mu, p_i_lv)

        z_i = self._sample(q_i_mu, q_i_lv, self.training)
        z_s = self._sample(q_s_mu, q_s_lv, self.training)
        z = torch.cat([z_i, z_s], dim=-1)
        x_hat = self.decoder(z)
        C = self.capacity_schedule(global_step=global_step, C_max=C_max, C_stop_iter=C_stop_iter)
        return {
            "z_i": z_i,
            "z_s": z_s,
            "z": z,
            "x_hat": x_hat,
            "C": C,
            "posterior_params": {
                "i": (q_i_mu, q_i_lv),
                "s": (q_s_mu, q_s_lv),
            },
            "prior_params": prior_params,
        }

    def get_parameters(self, base_lr: float = 1.0) -> List[Dict[str, Any]]:
        return [{"params": self.parameters(), "lr": base_lr}]


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambda_grl)


class SubjectClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


def _kl_diag_gaussian(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
) -> torch.Tensor:
    """KL(q||p) for diagonal Gaussians, reduced as mean over batch."""
    kl = 0.5 * (
        p_logvar
        - q_logvar
        + (torch.exp(q_logvar) + (q_mu - p_mu).pow(2))
        / torch.exp(p_logvar).clamp(min=1e-8)
        - 1.0
    )
    return kl.sum(dim=-1).mean()


def scvae_loss(
    scvae_output: Dict[str, Any],
    recon_target: torch.Tensor,
    beta_s: float,
    beta_i: float,
    lambda_recon: float,
    lambda_cl: float,
    contrastive_loss_val: Optional[torch.Tensor] = None,
    subj_logits_zs: Optional[torch.Tensor] = None,
    subj_logits_zi_adv: Optional[torch.Tensor] = None,
    subj_labels: Optional[torch.Tensor] = None,
    lambda_subj_zs: float = 1.0,
    lambda_subj_zi_adv: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Loss: recon + capacity-ramped KL_s/KL_i + lambda_cl*CL + optional subject heads."""
    loss_recon = F.mse_loss(scvae_output["x_hat"], recon_target, reduction="mean")

    q_s_mu, q_s_lv = scvae_output["posterior_params"]["s"]
    p_s_mu, p_s_lv = scvae_output["prior_params"]["s"]
    kl_s_raw = _kl_diag_gaussian(q_s_mu, q_s_lv, p_s_mu, p_s_lv)

    q_i_mu, q_i_lv = scvae_output["posterior_params"]["i"]
    p_i = scvae_output.get("prior_params", {}).get("i")
    if p_i is None:
        zeros = torch.zeros_like(q_i_mu)
        p_i_mu, p_i_lv = zeros, zeros
    else:
        p_i_mu, p_i_lv = p_i
    kl_i_raw = _kl_diag_gaussian(q_i_mu, q_i_lv, p_i_mu, p_i_lv)

    C = float(scvae_output.get("C", 0.0))
    kl_s = beta_s * (kl_s_raw - C).abs()
    kl_i = beta_i * (kl_i_raw - C).abs()
    recon_weighted = lambda_recon * loss_recon
    total = recon_weighted + kl_s + kl_i
    components: Dict[str, torch.Tensor] = {
        "recon": loss_recon.detach(),
        "recon_weighted": recon_weighted.detach(),
        "kl_s": kl_s_raw.detach(),
        "kl_i": kl_i_raw.detach(),
        "kl_s_weighted": kl_s.detach(),
        "kl_i_weighted": kl_i.detach(),
        "C": torch.tensor(C, device=recon_target.device),
    }

    if contrastive_loss_val is not None:
        total = total + lambda_cl * contrastive_loss_val
        components["contrastive"] = contrastive_loss_val.detach()
        components["contrastive_weighted"] = (
            lambda_cl * contrastive_loss_val
        ).detach()

    if subj_logits_zs is not None and subj_labels is not None:
        loss_subj_zs = F.cross_entropy(subj_logits_zs, subj_labels)
        total = total + lambda_subj_zs * loss_subj_zs
        pred = torch.argmax(subj_logits_zs, dim=1)
        components["subj_ce_zs"] = loss_subj_zs.detach()
        components["subj_ce_zs_weighted"] = (
            lambda_subj_zs * loss_subj_zs
        ).detach()
        components["subj_acc_zs"] = (pred == subj_labels).float().mean().detach()

    if subj_logits_zi_adv is not None and subj_labels is not None:
        loss_subj_zi_adv = F.cross_entropy(subj_logits_zi_adv, subj_labels)
        total = total + lambda_subj_zi_adv * loss_subj_zi_adv
        pred = torch.argmax(subj_logits_zi_adv, dim=1)
        components["subj_ce_zi_adv"] = loss_subj_zi_adv.detach()
        components["subj_ce_zi_adv_weighted"] = (
            lambda_subj_zi_adv * loss_subj_zi_adv
        ).detach()
        components["subj_acc_zi_adv"] = (pred == subj_labels).float().mean().detach()

    components["total"] = total.detach()
    return total, components
