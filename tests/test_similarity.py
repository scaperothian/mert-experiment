"""Tests for cosine similarity matrices and print formatting."""

import io
import sys

import numpy as np
import torch

from mert_experiment.similarity import cosine_block, cosine_matrix, print_matrix


# --- cosine_matrix -----------------------------------------------------------

class TestCosineMatrix:
    def test_identical_vectors_give_one(self):
        v = torch.randn(4)
        v = v / v.norm()
        vectors = v.unsqueeze(0).expand(3, -1)
        M = cosine_matrix(vectors)
        assert np.allclose(M, np.ones((3, 3)), atol=1e-5)

    def test_orthogonal_vectors_give_zero(self):
        e1 = torch.tensor([1.0, 0.0, 0.0])
        e2 = torch.tensor([0.0, 1.0, 0.0])
        e3 = torch.tensor([0.0, 0.0, 1.0])
        vectors = torch.stack([e1, e2, e3])
        M = cosine_matrix(vectors)
        assert np.allclose(np.diag(M), 1.0, atol=1e-6)
        assert np.allclose(M - np.eye(3), 0.0, atol=1e-6)

    def test_diagonal_is_one_for_unit_vectors(self):
        torch.manual_seed(0)
        raw = torch.randn(5, 16)
        vectors = raw / raw.norm(dim=1, keepdim=True)
        M = cosine_matrix(vectors)
        assert np.allclose(np.diag(M), 1.0, atol=1e-5)

    def test_matrix_is_symmetric(self):
        torch.manual_seed(1)
        raw = torch.randn(6, 32)
        vectors = raw / raw.norm(dim=1, keepdim=True)
        M = cosine_matrix(vectors)
        assert np.allclose(M, M.T, atol=1e-6)

    def test_output_is_numpy(self):
        M = cosine_matrix(torch.eye(3))
        assert isinstance(M, np.ndarray)

    def test_output_shape(self):
        torch.manual_seed(2)
        N, D = 7, 64
        raw = torch.randn(N, D)
        vectors = raw / raw.norm(dim=1, keepdim=True)
        assert cosine_matrix(vectors).shape == (N, N)

    def test_values_bounded(self):
        torch.manual_seed(3)
        raw = torch.randn(10, 16)
        vectors = raw / raw.norm(dim=1, keepdim=True)
        M = cosine_matrix(vectors)
        assert M.min() >= -1.0 - 1e-5
        assert M.max() <= 1.0 + 1e-5

    def test_known_2d_values(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([1.0, 1.0]) / (2 ** 0.5)
        M = cosine_matrix(torch.stack([a, b]))
        assert abs(M[0, 1] - 2 ** 0.5 / 2) < 1e-5
        assert abs(M[1, 0] - 2 ** 0.5 / 2) < 1e-5


# --- cosine_block ------------------------------------------------------------

class TestCosineBlock:
    def _unit(self, n: int, dim: int = 16, seed: int = 0) -> torch.Tensor:
        torch.manual_seed(seed)
        raw = torch.randn(n, dim)
        return raw / raw.norm(dim=1, keepdim=True)

    def test_output_shape_rectangular(self):
        A = self._unit(3, seed=0)
        B = self._unit(7, seed=1)
        M = cosine_block(A, B)
        assert M.shape == (3, 7)

    def test_output_shape_square(self):
        A = self._unit(4, seed=2)
        M = cosine_block(A, A)
        assert M.shape == (4, 4)

    def test_self_block_diagonal_is_one(self):
        A = self._unit(5, seed=3)
        M = cosine_block(A, A)
        assert np.allclose(np.diag(M), 1.0, atol=1e-5)

    def test_output_is_numpy(self):
        A = self._unit(3, seed=4)
        B = self._unit(4, seed=5)
        assert isinstance(cosine_block(A, B), np.ndarray)

    def test_values_bounded(self):
        A = self._unit(6, seed=6)
        B = self._unit(8, seed=7)
        M = cosine_block(A, B)
        assert M.min() >= -1.0 - 1e-5
        assert M.max() <= 1.0 + 1e-5

    def test_orthogonal_rows_give_zero(self):
        A = torch.tensor([[1.0, 0.0]])
        B = torch.tensor([[0.0, 1.0]])
        M = cosine_block(A, B)
        assert abs(M[0, 0]) < 1e-6

    def test_known_value(self):
        # 45-degree angle -> cosine = sqrt(2)/2
        A = torch.tensor([[1.0, 0.0]])
        B = torch.tensor([[1.0, 1.0]]) / (2 ** 0.5)
        M = cosine_block(A, B)
        assert abs(M[0, 0] - 2 ** 0.5 / 2) < 1e-5

    def test_not_symmetric_for_different_inputs(self):
        A = self._unit(3, seed=8)
        B = self._unit(3, seed=9)
        M = cosine_block(A, B)
        assert not np.allclose(M, M.T, atol=1e-4)

    def test_agrees_with_cosine_matrix_for_square_case(self):
        A = self._unit(5, seed=10)
        assert np.allclose(cosine_block(A, A), cosine_matrix(A), atol=1e-5)


# --- print_matrix ------------------------------------------------------------

class TestPrintMatrix:
    def _capture(self, *args, **kwargs) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            print_matrix(*args, **kwargs)
        finally:
            sys.stdout = old
        return buf.getvalue()

    def test_row_count_square(self):
        out = self._capture(np.eye(3), ["A", "B", "C"])
        lines = [l for l in out.splitlines() if l.strip()]
        assert len(lines) == 4   # 1 header + 3 data rows

    def test_row_count_rectangular(self):
        M = np.ones((2, 4))
        out = self._capture(M, ["R0", "R1"], ["C0", "C1", "C2", "C3"])
        lines = [l for l in out.splitlines() if l.strip()]
        assert len(lines) == 3   # 1 header + 2 data rows

    def test_header_contains_col_labels(self):
        out = self._capture(np.eye(2), ["Verse", "Chorus"])
        header = out.splitlines()[0]
        assert "Verse" in header
        assert "Chorus" in header

    def test_separate_row_and_col_labels(self):
        M = np.array([[0.9, 0.1], [0.3, 0.8], [0.5, 0.6]])
        out = self._capture(M, ["sec0", "sec1", "sec2"], ["0.0s", "1.0s"])
        assert "sec0" in out
        assert "0.0s" in out

    def test_diagonal_values_present(self):
        out = self._capture(np.eye(3), ["X", "Y", "Z"])
        assert "1.000" in out
        assert "0.000" in out

    def test_mark_columns_adds_extra_header_row(self):
        M = np.ones((2, 3))
        out = self._capture(
            M, ["sec0", "sec1"], ["0s", "1s", "2s"],
            mark_columns=[0, 1, 0],
        )
        lines = [l for l in out.splitlines() if l.strip()]
        # marker row + header row + 2 data rows
        assert len(lines) == 4

    def test_mark_columns_content_appears(self):
        M = np.ones((1, 2))
        out = self._capture(M, ["sec0"], ["0s", "1s"], mark_columns=["A", "B"])
        assert "A" in out
        assert "B" in out

    def test_long_row_labels_do_not_crash(self):
        M = np.eye(2)
        labels = ["A" * 50, "B" * 50]
        out = self._capture(M, labels)
        assert len(out) > 0

    def test_custom_cell_width(self):
        M = np.array([[0.123456]])
        out = self._capture(M, ["r"], ["c"], cell_w=4)
        # value rounded to 3 decimal places within 4-char cell
        assert "0.123" in out

    def test_col_labels_default_to_row_labels(self):
        M = np.eye(2)
        labels = ["P", "Q"]
        out_default = self._capture(M, labels)
        out_explicit = self._capture(M, labels, labels)
        assert out_default == out_explicit
