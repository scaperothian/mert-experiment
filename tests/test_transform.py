"""Tests for mean-centering and ZCA whitening."""

import pytest
import torch
import numpy as np

from mert_experiment.transform import (
    WHITEN_EPS,
    apply_transform,
    fit_mu,
    fit_transform,
    fit_whitening_matrix,
    l2_normalize_rows,
)


# ---------------------------------------------------------------------------
# fit_mu
# ---------------------------------------------------------------------------

class TestFitMu:
    def test_shape(self):
        X = torch.randn(20, 16)
        mu = fit_mu(X)
        assert mu.shape == (1, 16)

    def test_equals_row_mean(self):
        X = torch.randn(10, 8)
        mu = fit_mu(X)
        assert torch.allclose(mu, X.mean(dim=0, keepdim=True), atol=1e-5)

    def test_single_row(self):
        X = torch.tensor([[1.0, 2.0, 3.0]])
        mu = fit_mu(X)
        assert torch.allclose(mu, X, atol=1e-6)

    def test_constant_rows_equal_that_value(self):
        v = torch.tensor([3.0, -1.0, 2.0])
        X = v.unsqueeze(0).expand(5, -1)
        mu = fit_mu(X)
        assert torch.allclose(mu, v.unsqueeze(0), atol=1e-6)


# ---------------------------------------------------------------------------
# fit_whitening_matrix
# ---------------------------------------------------------------------------

class TestFitWhiteningMatrix:
    def test_output_shape(self):
        X = torch.randn(50, 8)
        W = fit_whitening_matrix(X)
        assert W.shape == (8, 8)

    def test_approximately_decorrelates(self):
        # whitened covariance should be close to identity
        torch.manual_seed(0)
        n, d = 200, 8
        X = torch.randn(n, d)
        X = X - X.mean(dim=0)
        W = fit_whitening_matrix(X)
        Xw = X @ W
        cov = (Xw.T @ Xw) / (n - 1)
        assert torch.allclose(cov, torch.eye(d), atol=0.1)

    def test_eps_prevents_zero_division(self):
        # degenerate input: rank-1 matrix
        X = torch.ones(10, 4)
        W = fit_whitening_matrix(X, eps=1e-4)
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()

    def test_result_is_symmetric(self):
        torch.manual_seed(1)
        X = torch.randn(30, 6)
        X = X - X.mean(dim=0)
        W = fit_whitening_matrix(X)
        assert torch.allclose(W, W.T, atol=1e-4)


# ---------------------------------------------------------------------------
# apply_transform
# ---------------------------------------------------------------------------

class TestApplyTransform:
    def test_center_only_subtracts_mu(self):
        X = torch.randn(10, 8)
        mu = torch.ones(1, 8)
        out = apply_transform(X, mu, None)
        assert torch.allclose(out, X - 1.0, atol=1e-6)

    def test_no_transform_returns_x_minus_zeros(self):
        X = torch.randn(5, 4)
        mu = torch.zeros(1, 4)
        out = apply_transform(X, mu, None)
        assert torch.allclose(out, X, atol=1e-6)

    def test_whiten_changes_values(self):
        torch.manual_seed(2)
        X = torch.randn(20, 6)
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        W = fit_whitening_matrix(Xc)
        out = apply_transform(X, mu, W)
        # output should differ from centered-only
        assert not torch.allclose(out, Xc, atol=1e-3)

    def test_output_shape_preserved(self):
        X = torch.randn(7, 12)
        mu = torch.zeros(1, 12)
        out = apply_transform(X, mu, None)
        assert out.shape == X.shape

    def test_output_not_normalized(self):
        # apply_transform must NOT L2-normalize (caller's responsibility)
        X = torch.randn(5, 4) * 10
        mu = torch.zeros(1, 4)
        out = apply_transform(X, mu, None)
        norms = out.norm(dim=1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=0.1)


# ---------------------------------------------------------------------------
# l2_normalize_rows
# ---------------------------------------------------------------------------

class TestL2NormalizeRows:
    def test_all_rows_unit_norm(self):
        torch.manual_seed(3)
        X = torch.randn(10, 16)
        out = l2_normalize_rows(X)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_shape_preserved(self):
        X = torch.randn(7, 5)
        assert l2_normalize_rows(X).shape == X.shape

    def test_zero_row_does_not_nan(self):
        X = torch.zeros(3, 4)
        out = l2_normalize_rows(X)
        assert not torch.isnan(out).any()

    def test_already_unit_rows_unchanged(self):
        torch.manual_seed(4)
        raw = torch.randn(5, 8)
        unit = raw / raw.norm(dim=1, keepdim=True)
        out = l2_normalize_rows(unit)
        assert torch.allclose(out, unit, atol=1e-5)


# ---------------------------------------------------------------------------
# fit_transform
# ---------------------------------------------------------------------------

class TestFitTransform:
    def _frames(self, n=50, d=16, seed=0):
        torch.manual_seed(seed)
        return torch.randn(n, d)

    def test_center_only_returns_none_for_W(self):
        mu, W = fit_transform(self._frames(), center=True, whiten=False)
        assert W is None
        assert mu.shape == (1, 16)

    def test_whiten_returns_matrix(self):
        mu, W = fit_transform(self._frames(), center=True, whiten=True)
        assert W is not None
        assert W.shape == (16, 16)

    def test_no_center_no_whiten_mu_is_zeros(self):
        mu, W = fit_transform(self._frames(), center=False, whiten=False)
        assert torch.allclose(mu, torch.zeros(1, 16))
        assert W is None

    def test_whiten_without_center_raises(self):
        with pytest.raises(ValueError, match="center"):
            fit_transform(self._frames(), center=False, whiten=True)

    def test_mu_equals_fit_mu(self):
        X = self._frames()
        mu_direct = X.mean(dim=0, keepdim=True)
        mu, _ = fit_transform(X, center=True, whiten=False)
        assert torch.allclose(mu, mu_direct, atol=1e-6)

    def test_center_and_whiten_mu_shape(self):
        X = self._frames(n=30, d=8, seed=5)
        mu, W = fit_transform(X, center=True, whiten=True)
        assert mu.shape == (1, 8)
        assert W.shape == (8, 8)

    def test_pipeline_produces_unit_norm_frames(self):
        X = self._frames()
        mu, W = fit_transform(X, center=True, whiten=True)
        F = l2_normalize_rows(apply_transform(X, mu, W))
        norms = F.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
