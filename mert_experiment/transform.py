"""
Mean-centering and ZCA whitening for MERT embeddings.

Pretrained audio SSL embeddings are anisotropic: most variance sits in a
narrow cone, so unrelated items end up with uniformly high cosine similarity
(e.g. 0.85–0.95) even when musically distinct. Two standard fixes:

  center   subtract the song-mean from every embedding before normalizing.
           Removes the dominant "what this song sounds like overall" component.
           Usually the biggest single improvement.

  whiten   after centering, multiply by the inverse-square-root of the
           covariance matrix (ZCA whitening). Decorrelates dimensions so the
           residual variation is more discriminative. Requires centering.

Both transforms are fit ONCE from the full-song frame embeddings and applied
identically to section prototypes and live frames. The pipeline is:

    raw window-mean  ->  center  ->  (whiten)  ->  L2-normalize  ->  cosine

Prototypes are built by transforming each window first, then averaging the
transformed windows, then L2-normalizing — not by averaging raw windows and
only normalizing the prototype at the end (which would re-introduce the
song-level mean we subtracted).
"""

from __future__ import annotations

import torch

WHITEN_EPS = 1e-6


def fit_mu(frame_means: torch.Tensor) -> torch.Tensor:
    """
    Song-mean vector, shape [1, D].

    Fit from the full-song frame means (not from section-level data only).
    """
    return frame_means.mean(dim=0, keepdim=True)


def fit_whitening_matrix(
    frames_centered: torch.Tensor,
    eps: float = WHITEN_EPS,
) -> torch.Tensor:
    """
    ZCA whitening matrix fit on already-centered frames.

    Returns W [D, D] such that (X - mu) @ W has approximately identity
    covariance. SVD-based; low-variance dimensions are regularized by eps.

    Args:
        frames_centered: [N, D] frames after mean subtraction
        eps:             regularizer for near-zero singular values
    """
    n = frames_centered.shape[0]
    cov = (frames_centered.T @ frames_centered) / max(n - 1, 1)
    U, S, _ = torch.linalg.svd(cov)
    return U @ torch.diag(1.0 / torch.sqrt(S + eps)) @ U.T


def apply_transform(
    X: torch.Tensor,
    mu: torch.Tensor,
    W: torch.Tensor | None,
) -> torch.Tensor:
    """
    Center and optionally whiten a batch of embeddings.

    Args:
        X:  [N, D] raw window means (NOT yet L2-normalized)
        mu: [1, D] song-mean from fit_mu  (or zeros to skip centering)
        W:  [D, D] whitening matrix from fit_whitening_matrix, or None

    Returns:
        [N, D] transformed embeddings — L2-normalization is intentionally
        left to the caller so it happens after whitening.
    """
    Xc = X - mu
    if W is not None:
        Xc = Xc @ W
    return Xc


def l2_normalize_rows(X: torch.Tensor) -> torch.Tensor:
    """L2-normalize each row. Returns [N, D]."""
    return X / (X.norm(dim=1, keepdim=True) + 1e-9)


def fit_transform(
    frame_means: torch.Tensor,
    center: bool = True,
    whiten: bool = True,
    eps: float = WHITEN_EPS,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Fit centering and/or whitening parameters from full-song frame means.

    Args:
        frame_means: [N_frames, D] raw (un-normalized) window means
        center:      compute and return a song-mean vector
        whiten:      compute a ZCA whitening matrix (requires center=True)
        eps:         SVD regularizer

    Returns:
        mu: [1, D]        centering mean (zero tensor if center=False)
        W:  [D, D] | None whitening matrix (None if whiten=False)

    Raises:
        ValueError if whiten=True and center=False.
    """
    if whiten and not center:
        raise ValueError("whiten=True requires center=True.")

    mu = fit_mu(frame_means) if center else torch.zeros(1, frame_means.shape[1])
    W = fit_whitening_matrix(frame_means - mu, eps=eps) if whiten else None
    return mu, W
