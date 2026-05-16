"""Sliding-window aggregation over MERT frame embeddings."""

from typing import Iterator

import torch

from .config import MERT_FRAME_RATE, WINDOW_SEC, HOP_SEC


def section_windows(
    start_sec: float,
    stop_sec: float,
    window: float = WINDOW_SEC,
    hop: float = HOP_SEC,
    fps: int = MERT_FRAME_RATE,
) -> Iterator[tuple[int, int]]:
    """
    Yield (start_frame, end_frame) pairs that tile [start_sec, stop_sec).

    A trailing window is added when the section is long enough but the last
    hop would otherwise miss the final frames.
    """
    win_frames = int(round(window * fps))
    hop_frames = int(round(hop * fps))
    start_f = int(round(start_sec * fps))
    stop_f  = int(round(stop_sec  * fps))

    f = start_f
    while f + win_frames <= stop_f:
        yield f, f + win_frames
        f += hop_frames

    # tail window: don't drop the last < hop seconds
    if (
        (stop_f - start_f) >= win_frames
        and (stop_f - win_frames) > (f - hop_frames)
    ):
        yield stop_f - win_frames, stop_f


def whole_song_window_spans(
    total_frames: int,
    window: float = WINDOW_SEC,
    hop: float = HOP_SEC,
    fps: int = MERT_FRAME_RATE,
) -> list[tuple[int, int]]:
    """
    Sliding windows covering [0, total_frames).

    No tail window — the final partial hop is intentionally dropped so every
    span is exactly win_frames wide. Used to simulate the stream of 'live'
    audio frames fed against pre-built section prototypes.
    """
    win_f = int(round(window * fps))
    hop_f = int(round(hop * fps))
    spans: list[tuple[int, int]] = []
    f = 0
    while f + win_f <= total_frames:
        spans.append((f, f + win_f))
        f += hop_f
    return spans


def window_means(
    layer_frames: torch.Tensor,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Mean-pool each span of frames WITHOUT L2 normalisation.

    Use this when embeddings will be centered or whitened before normalizing;
    normalization must happen after those transforms, not before.
    For pre-normalized embeddings use window_embeddings instead.

    Args:
        layer_frames: [T, D] frame embeddings for one MERT layer
        spans:        list of (start_frame, end_frame) pairs

    Returns:
        [W, D] raw (un-normalized) window means, or empty [0, D] if no valid spans
    """
    rows: list[torch.Tensor] = []
    T = layer_frames.shape[0]
    for s, e in spans:
        e = min(e, T)
        if e - s < 2:
            continue
        rows.append(layer_frames[s:e].mean(dim=0))
    if rows:
        return torch.stack(rows, dim=0)
    return torch.empty(0, layer_frames.shape[-1])


def window_embeddings(
    layer_frames: torch.Tensor,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Mean-pool and L2-normalise each span of frames.

    Clips each span end to the tensor length; skips spans shorter than 2 frames.

    Args:
        layer_frames: [T, D] frame embeddings for one MERT layer
        spans:        list of (start_frame, end_frame) pairs

    Returns:
        [W, D] L2-normalised embeddings, or empty [0, D] if no valid spans
    """
    rows: list[torch.Tensor] = []
    T = layer_frames.shape[0]
    for s, e in spans:
        e = min(e, T)
        if e - s < 2:
            continue
        v = layer_frames[s:e].mean(dim=0)
        v = v / (v.norm() + 1e-9)
        rows.append(v)
    if rows:
        return torch.stack(rows, dim=0)
    return torch.empty(0, layer_frames.shape[-1])


def pool_normalized(W: torch.Tensor) -> torch.Tensor:
    """
    Mean of L2-normalised window vectors, then re-normalised.

    Args:
        W: [num_windows, D]

    Returns:
        [D] unit-norm prototype vector
    """
    v = W.mean(dim=0)
    return v / (v.norm() + 1e-9)


def pool_section(
    layer_frames: torch.Tensor,
    start_sec: float,
    stop_sec: float,
    window: float = WINDOW_SEC,
    hop: float = HOP_SEC,
    fps: int = MERT_FRAME_RATE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a section prototype from its sliding-window embeddings.

    Args:
        layer_frames: [T, D] frame embeddings for one MERT layer
        start_sec, stop_sec: section boundaries in seconds

    Returns:
        windows: [num_windows, D]  L2-normalised per window
        pooled:  [D]               pool_normalized of windows
    """
    spans = list(section_windows(start_sec, stop_sec, window=window, hop=hop, fps=fps))
    W = window_embeddings(layer_frames, spans)

    if W.shape[0] == 0:
        raise ValueError(
            f"Section {start_sec:.2f}-{stop_sec:.2f}s is too short to form "
            f"even one window of {window}s at {fps} fps."
        )

    return W, pool_normalized(W)


def frame_to_section_assignment(
    frame_span: tuple[int, int],
    sections: list[dict],
    fps: int = MERT_FRAME_RATE,
) -> int:
    """
    Return the index of the section whose bounds contain the frame's centre time.

    Args:
        frame_span: (start_frame, end_frame)
        sections:   list of dicts with 'start' and 'stop' keys (seconds)

    Returns:
        Section index, or -1 if the centre falls outside all sections.
    """
    center_sec = (frame_span[0] + frame_span[1]) / 2 / fps
    for i, sec in enumerate(sections):
        if sec["start"] <= center_sec < sec["stop"]:
            return i
    return -1
