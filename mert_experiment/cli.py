"""
mert-sim: section prototype vs. all-frame similarity (VERSION A).

For each probed MERT layer:
  1. Build a pooled 'gold' prototype embedding per section from the clean recording.
  2. Slide windows across the whole song to produce N_frames embeddings.
  3. Report argmax accuracy: how often the highest-similarity prototype matches
     the ground-truth section for that frame.
  4. Plot similarity scores as a line chart over time (enabled by default).

In live deployment the same logic applies: prototypes stay fixed as gold
references; incoming audio frames stream in and are compared against each.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import LAYERS_TO_PROBE, MERT_FRAME_RATE, TARGET_SR
from .embed import embed_full_song, load_model
from .io import load_audio, load_song
from .plotting import _pairwise_path, plot_prototype_similarity, plot_section_grids
from .similarity import cosine_block, cosine_matrix
from .windows import (
    frame_to_section_assignment,
    pool_normalized,
    whole_song_window_spans,
    window_embeddings,
    section_windows,
)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

def analyze_layer(
    frames: torch.Tensor,
    sections: list[dict],
    all_spans: list[tuple[int, int]],
    section_of: list[int],
    layer_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build prototypes and frame embeddings for one MERT layer.

    Returns:
        A: [N_sec, N_frames]  prototype-vs-frame cosine similarity
        P: [N_sec, D]         section prototype embeddings (numpy)
        F: [N_frames, D]      all-frame embeddings (numpy)
    """
    prototypes = []
    for sec in sections:
        spans = list(section_windows(sec["start"], sec["stop"]))
        W = window_embeddings(frames, spans)
        prototypes.append(pool_normalized(W))
    P_t = torch.stack(prototypes)                  # [N_sec, D]
    F_t = window_embeddings(frames, all_spans)     # [N_frames, D]
    A = cosine_block(P_t, F_t)                    # [N_sec, N_frames]

    argmax = A.argmax(axis=0)
    correct = sum(1 for k, gt in enumerate(section_of) if gt >= 0 and argmax[k] == gt)
    total = sum(1 for gt in section_of if gt >= 0)
    acc = f"{correct}/{total} = {correct / total:.1%}" if total else "n/a"
    print(f"  Layer {layer_idx}: {A.shape[0]} prototypes × {A.shape[1]} frames  "
          f"| argmax accuracy: {acc}")

    return A, P_t.numpy(), F_t.numpy()


# ---------------------------------------------------------------------------
# Shared analysis core
# ---------------------------------------------------------------------------

def _run(
    audio_path: Path,
    sections: list[dict],
    device: str,
    also_pairwise: bool,
    plot: bool,
    plot_output: Path | None,
    save_npz: Path | None,
    audio_name: str,
) -> None:
    processor, model = load_model(device)
    wav = load_audio(audio_path)
    duration = wav.shape[0] / TARGET_SR
    print(f"  {duration:.2f}s, {wav.shape[0]} samples @ {TARGET_SR} Hz")

    print("Running MERT inference...")
    hidden = embed_full_song(wav, model, processor, device)
    total_frames = hidden.shape[1]
    print(f"  hidden states: {tuple(hidden.shape)}  (layers+1, frames, dim)")

    all_spans  = whole_song_window_spans(total_frames)
    section_of = [frame_to_section_assignment(sp, sections) for sp in all_spans]
    timestamps = np.array([(s + e) / 2 / MERT_FRAME_RATE for s, e in all_spans])

    print("\nAnalysing layers:")
    layer_A:   dict[int, np.ndarray] = {}
    layer_P:   dict[int, np.ndarray] = {}
    layer_F:   dict[int, np.ndarray] = {}

    for layer_idx in LAYERS_TO_PROBE:
        if layer_idx >= hidden.shape[0]:
            continue
        A, P, F = analyze_layer(hidden[layer_idx], sections, all_spans, section_of, layer_idx)
        layer_A[layer_idx] = A
        layer_P[layer_idx] = P
        layer_F[layer_idx] = F

    # --- Prototype-vs-frame line plot ----------------------------------------
    if plot:
        layer_results = {k: (layer_A[k], timestamps) for k in layer_A}
        plot_prototype_similarity(layer_results, sections, audio_name, plot_output)

    # --- Section×section grid ------------------------------------------------
    if also_pairwise:
        section_labels = [s["label"][:20] for s in sections]
        layer_grids = {
            k: cosine_matrix(torch.from_numpy(layer_P[k]))
            for k in layer_P
        }
        if plot:
            grid_output = _pairwise_path(plot_output) if plot_output else None
            plot_section_grids(layer_grids, section_labels, audio_name, grid_output)
        if save_npz:
            pairwise_npz = _pairwise_path(save_npz)
            np.savez(str(pairwise_npz), **{
                f"layer{k}": v for k, v in layer_grids.items()
            })
            print(f"Pairwise matrix saved to {pairwise_npz}")

    # --- Save VERSION A npz --------------------------------------------------
    if save_npz:
        flat: dict = {
            "section_starts": np.array([s["start"] for s in sections]),
            "section_stops":  np.array([s["stop"]  for s in sections]),
            "timestamps":     timestamps,
        }
        for k in layer_A:
            flat[f"layer{k}_A"] = layer_A[k]
            flat[f"layer{k}_P"] = layer_P[k]
            flat[f"layer{k}_F"] = layer_F[k]
        np.savez(str(save_npz), **flat)
        print(f"Results saved to {save_npz}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mert-sim",
        description=(
            "Section prototype vs. all-frame similarity via MERT embeddings. "
            "Plots similarity scores over time by default."
        ),
    )
    p.add_argument(
        "input",
        help="Path to a ProPresenter JSON manifest OR an audio file (.wav, .mp3, …).",
    )
    p.add_argument(
        "--section",
        metavar=("LABEL", "START", "STOP"),
        nargs=3,
        action="append",
        dest="sections",
        help=(
            "Audio-file mode only: define a section as LABEL START STOP (seconds). "
            "Repeatable. Omit to treat the full file as one section."
        ),
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress all matplotlib output.",
    )
    p.add_argument(
        "--plot-output",
        default=None,
        metavar="FILE",
        help=(
            "Save the similarity line plot to FILE instead of displaying it. "
            "When --also-pairwise is set, the section grid is saved alongside "
            "with a '_pairwise' suffix (e.g. out.png → out_pairwise.png)."
        ),
    )
    p.add_argument(
        "--also-pairwise",
        action="store_true",
        help=(
            "Compute the pooled section×section similarity grid and plot it. "
            "If --save-npz is set, also saves the grid as a separate _pairwise.npz."
        ),
    )
    p.add_argument(
        "--save-npz",
        default=None,
        metavar="FILE",
        help="Save VERSION A matrices and embeddings to a .npz file.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if input_path.suffix.lower() == ".json":
        if args.sections:
            parser.error("--section is only valid when passing an audio file, not a JSON.")
        print(f"\nLoading MERT on {device}...")
        audio_path, sections = load_song(input_path)
        audio_name = input_path.stem

    elif input_path.suffix.lower() in AUDIO_SUFFIXES:
        print(f"\nLoading MERT on {device}...")
        if args.sections:
            sections = [
                {"label": s[0], "start": float(s[1]), "stop": float(s[2])}
                for s in args.sections
            ]
        else:
            import torchaudio
            info = torchaudio.info(str(input_path))
            duration = info.num_frames / info.sample_rate
            sections = [{"label": "full", "start": 0.0, "stop": duration}]
        audio_path = input_path
        audio_name = input_path.stem

    else:
        parser.error(
            f"Unrecognised input '{input_path}'. "
            f"Expected a .json or an audio file ({', '.join(sorted(AUDIO_SUFFIXES))})."
        )
        return

    print(f"Audio: {audio_path}")
    print(f"Sections ({len(sections)}):")
    for i, s in enumerate(sections):
        print(f"  [{i}] {s['start']:6.2f}-{s['stop']:6.2f}s :: {s['label']}")
    print()

    _run(
        audio_path=audio_path,
        sections=sections,
        device=device,
        also_pairwise=args.also_pairwise,
        plot=not args.no_plot,
        plot_output=Path(args.plot_output) if args.plot_output else None,
        save_npz=Path(args.save_npz) if args.save_npz else None,
        audio_name=audio_name,
    )


if __name__ == "__main__":
    main()
