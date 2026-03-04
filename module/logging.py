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


def format_loss_breakdown(
    tag: str,
    avg_comp: Dict[str, float],
    ivae_enabled: bool,
    use_color: bool = False,
) -> str:
    """Return a compact one-line loss breakdown for console logging."""
    if not use_color:
        if not ivae_enabled:
            return f"{tag} total={avg_comp.get('total', 0.0):.4f}"

        return (
            f"{tag} total={avg_comp.get('total', 0.0):.4f} "
            f"rec={avg_comp.get('recon', 0.0):.4f}->{avg_comp.get('recon_weighted', avg_comp.get('recon', 0.0)):.4f} "
            f"KL[s/is/i/n]={avg_comp.get('kl_s', 0.0):.3f}/{avg_comp.get('kl_is', 0.0):.3f}/{avg_comp.get('kl_i', 0.0):.3f}/{avg_comp.get('kl_n', 0.0):.3f} "
            f"WKL[s/is/i/n]={avg_comp.get('kl_s_weighted', 0.0):.3f}/{avg_comp.get('kl_is_weighted', 0.0):.3f}/{avg_comp.get('kl_i_weighted', 0.0):.3f}/{avg_comp.get('kl_n_weighted', 0.0):.3f} "
            f"C={avg_comp.get('C', 0.0):.3f} "
            f"CL={avg_comp.get('contrastive', 0.0):.3f}->{avg_comp.get('contrastive_weighted', 0.0):.3f} "
            f"SubjCE[cls]={avg_comp.get('subj_ce_cls', 0.0):.3f}->{avg_comp.get('subj_ce_cls_weighted', 0.0):.3f} "
            f"AdvCE[adv]={avg_comp.get('subj_ce_adv', 0.0):.3f}->{avg_comp.get('subj_ce_adv_weighted', 0.0):.3f} "
            f"SubjAcc[cls/adv]={avg_comp.get('subj_acc_cls', 0.0):.3f}/{avg_comp.get('subj_acc_adv', 0.0):.3f}"
        )

    ansi_reset = "\033[0m"
    if "[test]" in tag.lower():
        c_tag = "\033[1;35m"  # bright magenta
        c_key = "\033[35m"    # magenta
        c_val = "\033[1;33m"  # bright yellow
    else:
        c_tag = "\033[1;36m"  # bright cyan
        c_key = "\033[36m"    # cyan
        c_val = "\033[1;32m"  # bright green

    def kv(key: str, val: str) -> str:
        return f"{c_key}{key}{ansi_reset}{c_val}{val}{ansi_reset}"

    if not ivae_enabled:
        total_val = f"{avg_comp.get('total', 0.0):.4f}"
        return f"{c_tag}{tag}{ansi_reset} {kv('total=', total_val)}"

    total = f"{avg_comp.get('total', 0.0):.4f}"
    rec = f"{avg_comp.get('recon', 0.0):.4f}->{avg_comp.get('recon_weighted', avg_comp.get('recon', 0.0)):.4f}"
    kl = f"{avg_comp.get('kl_s', 0.0):.3f}/{avg_comp.get('kl_is', 0.0):.3f}/{avg_comp.get('kl_i', 0.0):.3f}/{avg_comp.get('kl_n', 0.0):.3f}"
    wkl = f"{avg_comp.get('kl_s_weighted', 0.0):.3f}/{avg_comp.get('kl_is_weighted', 0.0):.3f}/{avg_comp.get('kl_i_weighted', 0.0):.3f}/{avg_comp.get('kl_n_weighted', 0.0):.3f}"
    capacity_str = f"{avg_comp.get('C', 0.0):.3f}"
    cl = f"{avg_comp.get('contrastive', 0.0):.3f}->{avg_comp.get('contrastive_weighted', 0.0):.3f}"
    subj_ce_cls = f"{avg_comp.get('subj_ce_cls', 0.0):.3f}->{avg_comp.get('subj_ce_cls_weighted', 0.0):.3f}"
    subj_ce_adv = f"{avg_comp.get('subj_ce_adv', 0.0):.3f}->{avg_comp.get('subj_ce_adv_weighted', 0.0):.3f}"
    subj_acc = f"{avg_comp.get('subj_acc_cls', 0.0):.3f}/{avg_comp.get('subj_acc_adv', 0.0):.3f}"

    return (
        f"{c_tag}{tag}{ansi_reset} "
        f"{kv('total=', total)} "
        f"{kv('rec=', rec)} "
        f"{kv('KL[s/is/i/n]=', kl)} "
        f"{kv('WKL[s/is/i/n]=', wkl)} "
        f"{kv('C=', capacity_str)} "
        f"{kv('CL=', cl)} "
        f"{kv('SubjCE[cls]=', subj_ce_cls)} "
        f"{kv('AdvCE[adv]=', subj_ce_adv)} "
        f"{kv('SubjAcc[cls/adv]=', subj_acc)}"
    )


def write_component_scalars(writer, split: str, avg_comp: Dict[str, float], epoch: int) -> None:
    """Write averaged loss components to TensorBoard."""
    for k, v in avg_comp.items():
        writer.add_scalar(f"LossComponents/{split}/{k}", v, epoch)
