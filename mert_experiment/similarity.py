"""Cosine similarity matrices and terminal reporting."""

import numpy as np
import torch


def cosine_matrix(vectors: torch.Tensor) -> np.ndarray:
    """
    Square pairwise cosine similarity matrix.

    Args:
        vectors: [N, D] L2-normalised row vectors

    Returns:
        [N, N] numpy float array
    """
    return (vectors @ vectors.T).numpy()


def cosine_block(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    """
    Rectangular cosine similarity matrix between two sets of L2-normalised vectors.

    Args:
        A: [M, D] L2-normalised row vectors  (e.g. section prototypes)
        B: [N, D] L2-normalised row vectors  (e.g. all-song frame embeddings)

    Returns:
        [M, N] numpy float array
    """
    return (A @ B.T).numpy()


def print_matrix(
    M: np.ndarray,
    row_labels: list[str],
    col_labels: list[str] | None = None,
    cell_w: int = 7,
    mark_columns: list | None = None,
) -> None:
    """
    Pretty-print a similarity matrix with row and column labels.

    Args:
        M:            [R, C] numpy array
        row_labels:   labels for each row
        col_labels:   labels for each column; defaults to row_labels (square case)
        cell_w:       character width of each data cell
        mark_columns: optional extra header row printed above col_labels,
                      e.g. to show which section each frame column belongs to
    """
    if col_labels is None:
        col_labels = row_labels

    label_w = max(10, max(len(l) for l in row_labels) + 1)

    if mark_columns is not None:
        marker = " " * label_w + "".join(
            f"{str(mc):>{cell_w + 1}}" for mc in mark_columns
        )
        print(marker)

    header = " " * label_w + "".join(
        f"{l[:cell_w]:>{cell_w + 1}}" for l in col_labels
    )
    print(header)

    for i, row in enumerate(M):
        print(
            f"{row_labels[i][:label_w - 1]:>{label_w}}"
            + "".join(f"{v:>{cell_w + 1}.3f}" for v in row)
        )
