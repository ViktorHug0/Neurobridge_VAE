"""Loss-curve plotting utilities (static PNG via matplotlib + interactive HTML via Plotly)."""

from __future__ import annotations

import os


# Loss keys produced by the iVAE training loop.
RAW_LOSS_KEYS: list[str] = [
    "total", "recon", "kl_s", "kl_is", "kl_i", "kl_n", "C", "contrastive", "subj_ce_cls", "subj_ce_adv",
]
SCALED_LOSS_KEYS: list[str] = [
    "total", "recon_weighted", "kl_s_weighted", "kl_is_weighted", "kl_i_weighted", "kl_n_weighted",
    "contrastive_weighted", "subj_ce_cls_weighted", "subj_ce_adv_weighted",
]

PNG_SUBDIR = "png_graphs"
HTML_SUBDIR = "html_graphs"
# HTML export is optional; default off to avoid extra deps/runtime issues.
ENABLE_HTML_DEFAULT = False


def _png_path(out_dir: str, filename: str) -> str:
    subdir = os.path.join(out_dir, PNG_SUBDIR)
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, filename)


def _html_path(out_dir: str, filename: str) -> str:
    subdir = os.path.join(out_dir, HTML_SUBDIR)
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, filename)


def _plot_png(
    train_history: dict[str, list[float]],
    test_history: dict[str, list[float]],
    epochs: list[int],
    keys: list[str],
    title: str,
    out_path: str,
    use_log_y: bool = False,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    any_line = False
    plotted_keys: list[str] = []
    # tab20 avoids early colour repeats for our key count.
    cmap = matplotlib.colormaps.get_cmap("tab20")
    key_to_color = {k: cmap(i / max(len(keys), 1)) for i, k in enumerate(keys)}

    for k in keys:
        tr = train_history.get(k)
        te = test_history.get(k)
        if tr is not None and len(tr) > 0:
            n = min(len(epochs), len(tr))
            tr_y = np.asarray(tr[:n], dtype=float)
            if use_log_y:
                tr_y = np.maximum(tr_y, 1e-8)
            ax.plot(epochs[:n], tr_y, color=key_to_color[k], linestyle="-", label=f"train/{k}")
            any_line = True
            if k not in plotted_keys:
                plotted_keys.append(k)
        if te is not None and len(te) > 0:
            n = min(len(epochs), len(te))
            te_y = np.asarray(te[:n], dtype=float)
            if use_log_y:
                te_y = np.maximum(te_y, 1e-8)
            ax.plot(epochs[:n], te_y, color=key_to_color[k], linestyle="--", label=f"test/{k}")
            any_line = True
            if k not in plotted_keys:
                plotted_keys.append(k)

    top1 = test_history.get("top1_acc")
    if top1 is not None and len(top1) > 0:
        n = min(len(epochs), len(top1))
        ax2 = ax.twinx()
        ax2.plot(epochs[:n], top1[:n], color="black", linewidth=2.0, linestyle="--", label="test/top1_acc(%)")
        ax2.set_ylabel("top1 acc (%)")
        ax2.set_ylim(0.0, 40.0)
        h2, l2 = ax2.get_legend_handles_labels()
    else:
        h2, l2 = [], []

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if use_log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    if any_line:
        # Build paired [train/key, test/key] legend columns.
        handles, labels = ax.get_legend_handles_labels()
        handle_map = {lab: h for h, lab in zip(handles, labels)}
        ordered_handles: list = []
        ordered_labels: list[str] = []
        for k in plotted_keys:
            tr_lab = f"train/{k}"
            te_lab = f"test/{k}"
            if tr_lab in handle_map:
                ordered_handles.append(handle_map[tr_lab])
                ordered_labels.append(tr_lab)
            else:
                ordered_handles.append(plt.Line2D([], [], color=key_to_color[k], linestyle="-", alpha=0.0))
                ordered_labels.append("")
            if te_lab in handle_map:
                ordered_handles.append(handle_map[te_lab])
                ordered_labels.append(te_lab)
            else:
                ordered_handles.append(plt.Line2D([], [], color=key_to_color[k], linestyle="--", alpha=0.0))
                ordered_labels.append("")
        ax.legend(ordered_handles + h2, ordered_labels + l2, fontsize=8, loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "No matching loss keys found", ha="center", va="center")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_html(
    train_history: dict[str, list[float]],
    test_history: dict[str, list[float]],
    epochs: list[int],
    keys: list[str],
    title: str,
    out_path: str,
    use_log_y: bool = False,
) -> str:
    """Interactive Plotly figure: click legend to toggle traces, scroll/box to zoom."""
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    palette = qualitative.Plotly + qualitative.D3 + qualitative.G10
    key_to_color = {k: palette[i % len(palette)] for i, k in enumerate(keys)}

    fig = go.Figure()
    has_top1 = False

    for k in keys:
        tr = train_history.get(k)
        te = test_history.get(k)
        color = key_to_color[k]
        if tr is not None and len(tr) > 0:
            n = min(len(epochs), len(tr))
            y_vals = [max(v, 1e-8) for v in tr[:n]] if use_log_y else list(tr[:n])
            fig.add_trace(go.Scatter(
                x=list(epochs[:n]), y=y_vals,
                mode="lines", name=f"train/{k}",
                line=dict(color=color, dash="solid"),
                legendgroup=k,
            ))
        if te is not None and len(te) > 0:
            n = min(len(epochs), len(te))
            y_vals = [max(v, 1e-8) for v in te[:n]] if use_log_y else list(te[:n])
            fig.add_trace(go.Scatter(
                x=list(epochs[:n]), y=y_vals,
                mode="lines", name=f"test/{k}",
                line=dict(color=color, dash="dash"),
                legendgroup=k,
            ))

    top1 = test_history.get("top1_acc")
    if top1 is not None and len(top1) > 0:
        has_top1 = True
        n = min(len(epochs), len(top1))
        fig.add_trace(go.Scatter(
            x=list(epochs[:n]), y=list(top1[:n]),
            mode="lines", name="test/top1_acc(%)",
            line=dict(color="black", dash="dot", width=2),
            yaxis="y2",
        ))

    layout_kwargs: dict = dict(
        title=title,
        xaxis=dict(title="epoch"),
        yaxis=dict(title="loss", type="log" if use_log_y else "linear"),
        legend=dict(groupclick="toggleitem"),
        hovermode="x unified",
        template="plotly_white",
    )
    if has_top1:
        layout_kwargs["yaxis2"] = dict(
            title="top1 acc (%)", overlaying="y", side="right", range=[0, 40],
        )
    fig.update_layout(**layout_kwargs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Write a self-contained HTML so it opens correctly offline.
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    return out_path


def save_loss_component_plots(
    train_history: dict[str, list[float]],
    test_history: dict[str, list[float]],
    epochs: list[int],
    out_dir: str,
) -> dict[str, str]:
    """Save raw/scaled loss curves as static PNGs (png_graphs/) and interactive HTMLs (html_graphs/)."""
    os.makedirs(out_dir, exist_ok=True)
    enable_html = os.getenv("ENABLE_HTML_PLOTS", "1" if ENABLE_HTML_DEFAULT else "0").lower() in (
        "1", "true", "yes", "on"
    )

    specs: list[tuple[str, list[str], str, bool]] = [
        ("raw",        RAW_LOSS_KEYS,    "Loss evolution (raw components)",              False),
        ("scaled",     SCALED_LOSS_KEYS, "Loss evolution (scaled components)",           False),
        ("raw_log",    RAW_LOSS_KEYS,    "Loss evolution (raw components, log-y)",       True),
        ("scaled_log", SCALED_LOSS_KEYS, "Loss evolution (scaled components, log-y)",   True),
    ]

    paths: dict[str, str] = {}
    for key, keys, title, log_y in specs:
        stem = f"loss_components_{key.replace('_log', '')}" + ("_log" if log_y else "")
        png_out = _png_path(out_dir, f"{stem}.png")
        paths[key] = _plot_png(train_history, test_history, epochs, keys, title, png_out, use_log_y=log_y)
        if enable_html:
            html_out = _html_path(out_dir, f"{stem}.html")
            try:
                paths[f"{key}_html"] = _plot_html(
                    train_history, test_history, epochs, keys, title, html_out, use_log_y=log_y
                )
            except Exception as exc:
                # Keep training results usable even when Plotly export is unavailable.
                print(f"[WARN] Failed to save interactive loss plot '{html_out}': {exc}")
                paths[f"{key}_html"] = ""
        else:
            paths[f"{key}_html"] = ""

    return paths


def save_subject_probe_plot(
    probe_history: dict[str, dict[str, list[float]]],
    epochs: list[int],
    out_dir: str,
) -> str:
    """Save probe train/val accuracy curves for z_s, z_i, z_is, z_n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "subject_probe.png")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    latent_order = ["z_s", "z_is", "z_i", "z_n"]
    color_map = {
        "z_s": "tab:blue",
        "z_is": "tab:orange",
        "z_i": "tab:green",
        "z_n": "tab:red",
    }
    any_line = False
    for latent in latent_order:
        data = probe_history.get(latent, {})
        train_acc = data.get("train_acc", [])
        val_acc = data.get("val_acc", [])
        color = color_map.get(latent, None)
        if train_acc:
            n = min(len(epochs), len(train_acc))
            ax.plot(
                epochs[:n],
                np.asarray(train_acc[:n], dtype=float) * 100.0,
                linestyle="--",
                linewidth=1.6,
                color=color,
                label=f"{latent}/train",
            )
            any_line = True
        if val_acc:
            n = min(len(epochs), len(val_acc))
            ax.plot(
                epochs[:n],
                np.asarray(val_acc[:n], dtype=float) * 100.0,
                linestyle="-",
                linewidth=2.0,
                color=color,
                label=f"{latent}/val",
            )
            any_line = True

    ax.set_title("Subject probe accuracy by latent block")
    ax.set_xlabel("epoch")
    ax.set_ylabel("classification accuracy (%)")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.3)
    if any_line:
        ax.legend(loc="best", ncol=2, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No subject probe history available", ha="center", va="center")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
