"""Matplotlib visualisations for MERT similarity analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_prototype_similarity(
    layer_results: dict[int, tuple[np.ndarray, np.ndarray]],
    sections: list[dict],
    audio_name: str = "",
    output_path: str | Path | None = None,
) -> None:
    """
    Line plot of section-prototype similarity scores over time.

    One subplot per MERT layer, stacked vertically with a shared time axis.
    Within each subplot:
      - One line per section prototype, with circle markers at each frame
      - Vertical dashed lines at every section boundary from the JSON

    Args:
        layer_results:  layer_idx -> (A [N_sec, N_frames], timestamps [N_frames])
        sections:       list of dicts with 'label', 'start', 'stop'
        audio_name:     used in the figure title
        output_path:    save to file if given; display interactively otherwise
    """
    import matplotlib.pyplot as plt

    layers = sorted(layer_results)
    n_layers = len(layers)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(
        n_layers, 1,
        figsize=(14, 3 * n_layers),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, layer_idx in zip(axes, layers):
        A, timestamps = layer_results[layer_idx]
        n_sec = A.shape[0]

        for i in range(n_sec):
            label = sections[i]["label"][:20] if i < len(sections) else f"sec{i}"
            ax.plot(
                timestamps, A[i],
                marker="o", markersize=3, linewidth=1,
                color=colors[i % len(colors)],
                label=label,
            )

        # Section boundary lines
        boundaries = {s["start"] for s in sections} | {sections[-1]["stop"]}
        for t in sorted(boundaries):
            ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_ylabel("Cosine similarity", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    title = (
        f"Section prototype similarity — {audio_name}"
        if audio_name else "Section prototype similarity"
    )
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, output_path)


def plot_section_grids(
    layer_matrices: dict[int, np.ndarray],
    section_labels: list[str],
    audio_name: str = "",
    output_path: str | Path | None = None,
) -> None:
    """
    Grid heatmap of pooled section×section cosine similarity, one panel per layer.

    Each cell shows a numeric score on a heat-coloured background (RdYlGn,
    −1 to 1). Panels are laid out in a single row.

    Args:
        layer_matrices: layer_idx -> [N_sec, N_sec] numpy array
        section_labels: label for each row/column
        audio_name:     used in the figure title
        output_path:    save to file if given; display interactively otherwise
    """
    import matplotlib.pyplot as plt

    layers = sorted(layer_matrices)
    n_layers = len(layers)
    n_sec = len(section_labels)
    cell_size = max(1.5, n_sec * 0.7)

    fig, axes = plt.subplots(
        1, n_layers,
        figsize=(cell_size * n_layers + 1, cell_size + 1.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, layer_idx in zip(axes, layers):
        M = layer_matrices[layer_idx]
        im = ax.imshow(M, vmin=-1.0, vmax=1.0, cmap="RdYlGn", aspect="equal")

        for i in range(n_sec):
            for j in range(n_sec):
                val = float(M[i, j])
                text_color = "white" if abs(val) > 0.7 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold",
                )

        short = [lb[:14] for lb in section_labels]
        ax.set_xticks(range(n_sec))
        ax.set_yticks(range(n_sec))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        plt.colorbar(im, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)

    title = (
        f"Section×section similarity — {audio_name}"
        if audio_name else "Section×section similarity"
    )
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, output_path)


def _pairwise_path(path: str | Path) -> Path:
    """Derive the pairwise output path by inserting '_pairwise' before the extension."""
    p = Path(path)
    return p.with_stem(p.stem + "_pairwise")


def _save_or_show(fig, output_path: str | Path | None) -> None:
    import matplotlib.pyplot as plt

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
        plt.close(fig)
    else:
        plt.show()
