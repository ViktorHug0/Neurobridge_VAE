
import torch
from bisect import bisect_right
from typing import List, Tuple, Dict, Any, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import itertools


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layer=1, hidden_dim=1024) -> None:
        super(MLP, self).__init__()
        # Simple feed-forward utility MLP used to parameterize small heads
        # (e.g., domain flow parameters in legacy codepaths) or classifiers.
        model = []
        model += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layer):
            model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Starts with a warm-up phase, then decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        warmup_factor (float): a float number :math:`k` between 0 and 1, the start learning rate of warmup phase
            will be set to :math:`k*initial\_lr`
        warmup_steps (int): number of warm-up steps.
        warmup_method (str): "constant" denotes a constant learning rate during warm-up phase and "linear" denotes a
            linear-increasing learning rate during warm-up phase.
        last_epoch (int): The index of last epoch. Default: -1.
    """

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
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
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


# =====================================================================
# EEG iVAE: Identifiable VAE for EEG Inter-Subject Decoding
# Based on "Variational Autoencoders and Nonlinear ICA: A Unifying
# Framework" (Khemakhem et al., 2020) with beta-VAE capacity ramping
# from "Understanding disentangling in beta-VAE" (Burgess et al., 2018).
#
# Disentangles EEG latent space into 4 blocks:
#   z_s  : subject-related     (conditioned on subject ID S)
#   z_i  : image-related       (conditioned on image features I)
#   z_is : interaction          (conditioned on S and I)
#   z_n  : noise / residual    (unconditional, standard Gaussian prior)
# =====================================================================


class EEGiVAE(nn.Module):
    """Identifiable VAE bottleneck for EEG inter-subject decoding.

    Sits between an existing EEG backbone encoder and the contrastive
    alignment loss.  The latent space is split into four blocks with
    separate conditional priors (iVAE-style) to encourage disentanglement
    of subject-specific, image-specific, interaction, and noise factors.

    Prior factorisation:
        p(z | S, I) = p(z_s | S) * p(z_i | I) * p(z_is | S, I) * p(z_n)

    Each prior/posterior is a diagonal Gaussian parameterised by small
    networks (embeddings for discrete S, MLPs for continuous I).
    """

    def __init__(
        self,
        feature_dim: int,
        image_feature_dim: int,
        n_subjects: int = 11,
        z_s_dim: int = 32,
        z_i_dim: int = 128,
        z_is_dim: int = 64,
        z_n_dim: int = 32,
        hidden_dim: int = 512,
        subj_emb_dim: int = 64,
        n_layers: int = 1,
        reconstruct_raw_eeg: bool = False,
        raw_eeg_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            feature_dim:       EEG backbone output dimension.
            image_feature_dim: Pre-extracted image feature dim (e.g. 1024).
            n_subjects:        Embedding table size.  Must be > max(subject_id)
                               so that IDs can index the table directly
                               (e.g. 11 for Things-EEG subjects 1-10).
            z_s_dim:           Subject latent block dimension.
            z_i_dim:           Image latent block dimension.
            z_is_dim:          Interaction latent block dimension.
            z_n_dim:           Noise latent block dimension.
            hidden_dim:        Hidden layer width for all MLPs.
            subj_emb_dim:      Shared subject embedding dimension.
            n_layers:          Number of *extra* hidden layers in MLPs
                               (0 -> 2 linear layers, 1 -> 3, etc.).
            reconstruct_raw_eeg: If True the decoder targets the raw
                               (flattened) EEG signal; otherwise it targets
                               the backbone embedding.
            raw_eeg_dim:       channels * time_points (required when
                               reconstruct_raw_eeg is True).
        """
        super(EEGiVAE, self).__init__()

        self.feature_dim = feature_dim
        self.image_feature_dim = image_feature_dim
        self.n_subjects = n_subjects
        self.z_s_dim = z_s_dim
        self.z_i_dim = z_i_dim
        self.z_is_dim = z_is_dim
        self.z_n_dim = z_n_dim
        self.z_total_dim = z_s_dim + z_i_dim + z_is_dim + z_n_dim
        self.hidden_dim = hidden_dim
        self.subj_emb_dim = subj_emb_dim
        self.reconstruct_raw_eeg = reconstruct_raw_eeg
        self.raw_eeg_dim = raw_eeg_dim

        if reconstruct_raw_eeg and raw_eeg_dim is None:
            raise ValueError(
                "raw_eeg_dim must be specified when reconstruct_raw_eeg=True")

        recon_dim = raw_eeg_dim if reconstruct_raw_eeg else feature_dim

        # ── Shared subject embedding (posteriors + interaction prior) ──
        self.subject_embedding = nn.Embedding(n_subjects, subj_emb_dim)

        # ── Conditional prior networks ──
        # Subject prior: direct lookup → (mu, logvar)
        self.subject_prior = nn.Embedding(n_subjects, 2 * z_s_dim)

        # Image prior: MLP(image_features) → (mu, logvar)
        self.image_prior = MLP(
            image_feature_dim, 2 * z_i_dim,
            n_layer=n_layers, hidden_dim=hidden_dim)

        # Interaction prior: MLP(concat(subj_emb, image_features)) → (mu, logvar)
        self.interaction_prior = MLP(
            subj_emb_dim + image_feature_dim, 2 * z_is_dim,
            n_layer=n_layers, hidden_dim=hidden_dim)

        # Noise prior: fixed N(0, 1) — no learnable parameters

        # ── Posterior encoder networks ──
        # q(z_s | X, S)
        self.posterior_z_s = MLP(
            feature_dim + subj_emb_dim, 2 * z_s_dim,
            n_layer=n_layers, hidden_dim=hidden_dim)

        # q(z_i | X, I)
        self.posterior_z_i = MLP(
            feature_dim + image_feature_dim, 2 * z_i_dim,
            n_layer=n_layers, hidden_dim=hidden_dim)

        # q(z_is | X, S, I)
        self.posterior_z_is = MLP(
            feature_dim + subj_emb_dim + image_feature_dim, 2 * z_is_dim,
            n_layer=n_layers, hidden_dim=hidden_dim)

        # q(z_n | X)
        self.posterior_z_n = MLP(
            feature_dim, 2 * z_n_dim,
            n_layer=n_layers, hidden_dim=hidden_dim)

        # ── Decoder ──
        self.decoder = nn.Sequential(
            nn.Linear(self.z_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, recon_dim),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Standard VAE reparameterization: z = mu + sigma * eps."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu

    @staticmethod
    def _split_mu_logvar(
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a (B, 2*D) tensor into mu (B, D) and logvar (B, D)."""
        d = params.shape[-1] // 2
        return params[..., :d], params[..., d:]

    @staticmethod
    def kl_divergence_gaussian(
        q_mu: torch.Tensor, q_logvar: torch.Tensor,
        p_mu: torch.Tensor, p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Analytical KL(q || p) for diagonal Gaussians.

        Returns a scalar: sum over latent dims, mean over the batch.
        """
        kl = 0.5 * (
            p_logvar - q_logvar
            + (torch.exp(q_logvar) + (q_mu - p_mu).pow(2))
              / torch.exp(p_logvar).clamp(min=1e-8)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    @staticmethod
    def capacity_schedule(
        global_step: int, C_max: float, C_stop_iter: int,
    ) -> float:
        """Linear capacity ramp from 0 to *C_max* over *C_stop_iter* steps."""
        if C_stop_iter <= 0:
            return C_max
        return min(C_max * global_step / C_stop_iter, C_max)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor,
        image_features: torch.Tensor,
    ) -> Dict[str, Any]:
        """Encode backbone features into 4 disentangled latent blocks.

        Args:
            x:              (B, feature_dim) backbone output.
            subject_ids:    (B,) integer subject IDs (used as embedding indices).
            image_features: (B, image_feature_dim) pre-extracted image features.

        Returns:
            dict with keys  z_s, z_i, z_is, z_n, z,
                            posterior_params, prior_params.
        """
        subj_emb = self.subject_embedding(subject_ids)  # (B, subj_emb_dim)

        # ── Posterior parameters ──
        q_s_mu, q_s_lv = self._split_mu_logvar(
            self.posterior_z_s(torch.cat([x, subj_emb], dim=1)))
        q_i_mu, q_i_lv = self._split_mu_logvar(
            self.posterior_z_i(torch.cat([x, image_features], dim=1)))
        q_is_mu, q_is_lv = self._split_mu_logvar(
            self.posterior_z_is(
                torch.cat([x, subj_emb, image_features], dim=1)))
        q_n_mu, q_n_lv = self._split_mu_logvar(
            self.posterior_z_n(x))

        # ── Prior parameters ──
        p_s_mu, p_s_lv = self._split_mu_logvar(
            self.subject_prior(subject_ids))
        p_i_mu, p_i_lv = self._split_mu_logvar(
            self.image_prior(image_features))
        p_is_mu, p_is_lv = self._split_mu_logvar(
            self.interaction_prior(
                torch.cat([subj_emb, image_features], dim=1)))
        # Noise prior: N(0, 1) → mu=0, logvar=0
        p_n_mu = torch.zeros_like(q_n_mu)
        p_n_lv = torch.zeros_like(q_n_lv)

        # ── Sample via reparameterization ──
        z_s = self.reparameterize(q_s_mu, q_s_lv)
        z_i = self.reparameterize(q_i_mu, q_i_lv)
        z_is = self.reparameterize(q_is_mu, q_is_lv)
        z_n = self.reparameterize(q_n_mu, q_n_lv)
        z = torch.cat([z_s, z_i, z_is, z_n], dim=1)

        return {
            'z_s': z_s, 'z_i': z_i, 'z_is': z_is, 'z_n': z_n, 'z': z,
            'posterior_params': {
                's':  (q_s_mu,  q_s_lv),
                'i':  (q_i_mu,  q_i_lv),
                'is': (q_is_mu, q_is_lv),
                'n':  (q_n_mu,  q_n_lv),
            },
            'prior_params': {
                's':  (p_s_mu,  p_s_lv),
                'i':  (p_i_mu,  p_i_lv),
                'is': (p_is_mu, p_is_lv),
                'n':  (p_n_mu,  p_n_lv),
            },
        }

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from the full concatenated latent vector."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor,
        image_features: torch.Tensor,
        global_step: int = 0,
        C_max: float = 25.0,
        C_stop_iter: int = 10000,
    ) -> Dict[str, Any]:
        """Full forward: encode → sample → decode → per-block KL.

        Returns dict with keys:
            z_s, z_i, z_is, z_n, z, x_hat,
            kl_losses  (dict  block_name → scalar),
            C          (current capacity value).
        """
        enc = self.encode(x, subject_ids, image_features)
        x_hat = self.decode(enc['z'])

        C = self.capacity_schedule(global_step, C_max, C_stop_iter)

        kl_losses: Dict[str, torch.Tensor] = {}
        for blk in ('s', 'i', 'is', 'n'):
            q_mu, q_lv = enc['posterior_params'][blk]
            p_mu, p_lv = enc['prior_params'][blk]
            kl_losses[blk] = self.kl_divergence_gaussian(
                q_mu, q_lv, p_mu, p_lv)

        return {
            'z_s': enc['z_s'],
            'z_i': enc['z_i'],
            'z_is': enc['z_is'],
            'z_n': enc['z_n'],
            'z': enc['z'],
            'x_hat': x_hat,
            'kl_losses': kl_losses,
            'C': C,
        }

    def get_parameters(self, base_lr: float = 1.0) -> List[Dict[str, Any]]:
        """Return parameter groups (useful for differential LR)."""
        return [{"params": self.parameters(), "lr": base_lr}]


# =====================================================================
# Loss helper
# =====================================================================


def ivae_loss(
    ivae_output: Dict[str, Any],
    recon_target: torch.Tensor,
    betas: Dict[str, float],
    gamma_cl: float = 1.0,
    contrastive_loss_val: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the full iVAE training objective.

    L = L_recon
      + sum_k  beta_k * |KL_k - C|          (capacity-ramped KL)
      + gamma  * L_contrastive               (alignment on z_i)

    Args:
        ivae_output:          dict returned by :meth:`EEGiVAE.forward`.
        recon_target:         tensor that ``x_hat`` should reconstruct.
        betas:                ``{'s': ..., 'i': ..., 'is': ..., 'n': ...}``
                              per-block beta weights.
        gamma_cl:             contrastive loss weight.
        contrastive_loss_val: pre-computed contrastive loss (may be None
                              to skip contrastive term).

    Returns:
        (total_loss, components_dict)  where *components_dict* stores
        detached scalars for logging.
    """
    # Reconstruction (MSE, mean-reduced)
    loss_recon = F.mse_loss(ivae_output['x_hat'], recon_target, reduction='mean')

    C = ivae_output['C']

    # Per-block KL with beta-VAE capacity ramping: beta * |KL - C|
    loss_kl_total = torch.tensor(0.0, device=recon_target.device)
    components: Dict[str, torch.Tensor] = {}
    for blk, kl_val in ivae_output['kl_losses'].items():
        beta = betas.get(blk, 1.0)
        kl_cap = beta * (kl_val - C).abs()
        loss_kl_total = loss_kl_total + kl_cap
        components[f'kl_{blk}'] = kl_val.detach()
        components[f'kl_{blk}_weighted'] = kl_cap.detach()

    total = loss_recon + loss_kl_total

    components['recon'] = loss_recon.detach()
    components['kl_total'] = loss_kl_total.detach()
    components['C'] = torch.tensor(C, device=recon_target.device)

    if contrastive_loss_val is not None:
        total = total + gamma_cl * contrastive_loss_val
        components['contrastive'] = contrastive_loss_val.detach()
        components['contrastive_weighted'] = (
            gamma_cl * contrastive_loss_val).detach()

    components['total'] = total.detach()
    return total, components
