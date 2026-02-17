from collections import defaultdict
from typing import Dict, Any

import torch


def init_component_sums() -> defaultdict:
    """Create an accumulator dict for scalar loss components."""
    return defaultdict(float)


def accumulate_components(acc: dict, comp: Dict[str, Any]) -> None:
    """Accumulate (possibly tensor) scalar components into a float dict."""
    for k, v in comp.items():
        if torch.is_tensor(v):
            acc[k] += float(v.detach().item())
        else:
            acc[k] += float(v)


def average_components(acc: dict, n: int) -> Dict[str, float]:
    """Average accumulated components over n steps."""
    if n <= 0:
        return {}
    return {k: v / n for k, v in acc.items()}


def format_loss_breakdown(tag: str, avg_comp: Dict[str, float], ivae_enabled: bool) -> str:
    """Return a compact one-line loss breakdown for console logging."""
    if not ivae_enabled:
        return f"{tag} total={avg_comp.get('total', 0.0):.4f}"

    return (
        f"{tag} total={avg_comp.get('total', 0.0):.4f} "
        f"rec={avg_comp.get('recon', 0.0):.4f} "
        f"KL[s/i/is/n]={avg_comp.get('kl_s', 0.0):.3f}/"
        f"{avg_comp.get('kl_i', 0.0):.3f}/"
        f"{avg_comp.get('kl_is', 0.0):.3f}/"
        f"{avg_comp.get('kl_n', 0.0):.3f} "
        f"WKL[s/i/is/n]={avg_comp.get('kl_s_weighted', 0.0):.3f}/"
        f"{avg_comp.get('kl_i_weighted', 0.0):.3f}/"
        f"{avg_comp.get('kl_is_weighted', 0.0):.3f}/"
        f"{avg_comp.get('kl_n_weighted', 0.0):.3f} "
        f"C={avg_comp.get('C', 0.0):.3f} "
        f"CL={avg_comp.get('contrastive', 0.0):.3f}->{avg_comp.get('contrastive_weighted', 0.0):.3f}"
    )


def write_component_scalars(writer, split: str, avg_comp: Dict[str, float], epoch: int) -> None:
    """Write averaged loss components to TensorBoard."""
    for k, v in avg_comp.items():
        writer.add_scalar(f"LossComponents/{split}/{k}", v, epoch)
