"""Tests for matplotlib plotting functions (no display, file output only)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mert_experiment.plotting import (
    _layer_mean_path,
    _pairwise_path,
    causal_rolling_mean,
    plot_layer_mean_similarity,
    plot_prototype_similarity,
    plot_section_grids,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sections(n: int = 3) -> list[dict]:
    step = 30.0
    return [
        {"label": f"Section {i}", "start": i * step, "stop": (i + 1) * step}
        for i in range(n)
    ]


def _layer_results(
    n_sec: int = 3,
    n_frames: int = 40,
    layers: tuple = (4, 8, 12),
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    timestamps = np.linspace(0.5, 89.5, n_frames)
    return {
        layer: (rng.uniform(0, 1, (n_sec, n_frames)).astype(np.float32), timestamps)
        for layer in layers
    }


def _layer_A(
    n_sec: int = 3,
    n_frames: int = 40,
    layers: tuple = (4, 8, 12),
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(2)
    return {
        layer: rng.uniform(-1, 1, (n_sec, n_frames)).astype(np.float32)
        for layer in layers
    }


def _layer_grids(
    n_sec: int = 3,
    layers: tuple = (4, 8, 12),
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(1)
    result = {}
    for layer in layers:
        M = rng.uniform(-1, 1, (n_sec, n_sec)).astype(np.float32)
        np.fill_diagonal(M, 1.0)
        result[layer] = M
    return result


# ---------------------------------------------------------------------------
# _pairwise_path
# ---------------------------------------------------------------------------

class TestPairwisePath:
    def test_inserts_pairwise_suffix(self):
        assert _pairwise_path("out.png") == Path("out_pairwise.png")

    def test_preserves_extension(self):
        p = _pairwise_path("results.npz")
        assert p.suffix == ".npz"
        assert "pairwise" in p.stem

    def test_works_with_path_object(self):
        p = _pairwise_path(Path("data/out.png"))
        assert p.name == "out_pairwise.png"

    def test_stem_only_no_extension(self):
        p = _pairwise_path("myfile")
        assert "pairwise" in str(p)


# ---------------------------------------------------------------------------
# plot_prototype_similarity
# ---------------------------------------------------------------------------

class TestPlotPrototypeSimilarity:
    def test_saves_to_file(self, tmp_path):
        out = tmp_path / "sim.png"
        plot_prototype_similarity(
            _layer_results(), _sections(), audio_name="test", output_path=out
        )
        assert out.exists() and out.stat().st_size > 0

    def test_single_layer(self, tmp_path):
        out = tmp_path / "single.png"
        plot_prototype_similarity(
            _layer_results(layers=(6,)), _sections(), output_path=out
        )
        assert out.exists()

    def test_five_layers(self, tmp_path):
        out = tmp_path / "five.png"
        plot_prototype_similarity(
            _layer_results(layers=(4, 6, 8, 10, 12)), _sections(), output_path=out
        )
        assert out.exists()

    def test_one_section(self, tmp_path):
        out = tmp_path / "onesec.png"
        plot_prototype_similarity(
            _layer_results(n_sec=1), _sections(n=1), output_path=out
        )
        assert out.exists()

    def test_no_output_leaves_figure_open(self):
        import matplotlib.pyplot as plt
        before = len(plt.get_fignums())
        with patch("matplotlib.pyplot.show"):
            plot_prototype_similarity(_layer_results(layers=(4,)), _sections())
        assert len(plt.get_fignums()) > before
        plt.close("all")

    def test_empty_audio_name(self, tmp_path):
        out = tmp_path / "notitle.png"
        plot_prototype_similarity(
            _layer_results(), _sections(), audio_name="", output_path=out
        )
        assert out.exists()

    def test_many_sections(self, tmp_path):
        out = tmp_path / "manysec.png"
        plot_prototype_similarity(
            _layer_results(n_sec=8), _sections(n=8), output_path=out
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_section_grids
# ---------------------------------------------------------------------------

class TestPlotSectionGrids:
    def _labels(self, n: int) -> list[str]:
        return [f"Sec {i}" for i in range(n)]

    def test_saves_to_file(self, tmp_path):
        out = tmp_path / "grid.png"
        plot_section_grids(_layer_grids(), self._labels(3), output_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_single_layer(self, tmp_path):
        out = tmp_path / "single_grid.png"
        plot_section_grids(_layer_grids(layers=(6,)), self._labels(3), output_path=out)
        assert out.exists()

    def test_two_sections(self, tmp_path):
        out = tmp_path / "twosec.png"
        plot_section_grids(_layer_grids(n_sec=2), self._labels(2), output_path=out)
        assert out.exists()

    def test_large_grid(self, tmp_path):
        out = tmp_path / "large.png"
        plot_section_grids(_layer_grids(n_sec=8), self._labels(8), output_path=out)
        assert out.exists()

    def test_no_output_leaves_figure_open(self):
        import matplotlib.pyplot as plt
        before = len(plt.get_fignums())
        plot_section_grids(_layer_grids(layers=(4,)), self._labels(3))
        assert len(plt.get_fignums()) > before
        plt.close("all")

    def test_audio_name_in_title(self, tmp_path):
        out = tmp_path / "titled.png"
        plot_section_grids(
            _layer_grids(), self._labels(3), audio_name="My Song", output_path=out
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# causal_rolling_mean
# ---------------------------------------------------------------------------

class TestCausalRollingMean:
    def test_k1_returns_copy(self):
        A = np.array([[1.0, 2.0, 3.0, 4.0]])
        out = causal_rolling_mean(A, k=1)
        np.testing.assert_array_equal(out, A)

    def test_k3_first_sample_unchanged(self):
        A = np.array([[10.0, 2.0, 2.0, 2.0]])
        out = causal_rolling_mean(A, k=3)
        assert out[0, 0] == pytest.approx(10.0)

    def test_k3_second_sample_is_mean_of_two(self):
        A = np.array([[4.0, 6.0, 0.0, 0.0]])
        out = causal_rolling_mean(A, k=3)
        assert out[0, 1] == pytest.approx(5.0)

    def test_k3_third_sample_is_mean_of_three(self):
        A = np.array([[3.0, 6.0, 9.0, 0.0]])
        out = causal_rolling_mean(A, k=3)
        assert out[0, 2] == pytest.approx(6.0)

    def test_output_shape_preserved(self):
        A = np.random.default_rng(0).random((4, 50))
        assert causal_rolling_mean(A, k=5).shape == A.shape

    def test_no_future_frames_used(self):
        # A spike at index 5 should not influence indices 0–4
        A = np.zeros((1, 10))
        A[0, 5] = 100.0
        out = causal_rolling_mean(A, k=3)
        assert np.all(out[0, :5] == 0.0)

    def test_constant_signal_unchanged(self):
        A = np.ones((2, 20)) * 7.0
        out = causal_rolling_mean(A, k=4)
        np.testing.assert_allclose(out, A, atol=1e-6)

    def test_larger_k_smooths_more(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((1, 100))
        out3 = causal_rolling_mean(A, k=3)
        out9 = causal_rolling_mean(A, k=9)
        # larger k → smaller variance in output
        assert out9.var() < out3.var()


# ---------------------------------------------------------------------------
# plot_layer_mean_similarity
# ---------------------------------------------------------------------------

class TestPlotLayerMeanSimilarity:
    def _timestamps(self, n_frames: int = 40) -> np.ndarray:
        return np.linspace(0.5, 89.5, n_frames)

    def test_saves_to_file(self, tmp_path):
        out = tmp_path / "lm.png"
        plot_layer_mean_similarity(
            _layer_A(), self._timestamps(), _sections(), output_path=out
        )
        assert out.exists() and out.stat().st_size > 0

    def test_empty_dict_does_not_raise(self, tmp_path):
        plot_layer_mean_similarity({}, self._timestamps(), _sections())

    def test_single_layer(self, tmp_path):
        out = tmp_path / "lm_single.png"
        plot_layer_mean_similarity(
            _layer_A(layers=(6,)), self._timestamps(), _sections(), output_path=out
        )
        assert out.exists()

    def test_custom_smooth_k(self, tmp_path):
        out = tmp_path / "lm_k5.png"
        plot_layer_mean_similarity(
            _layer_A(), self._timestamps(), _sections(), output_path=out, smooth_k=5
        )
        assert out.exists()

    def test_no_output_leaves_figure_open(self):
        import matplotlib.pyplot as plt
        before = len(plt.get_fignums())
        plot_layer_mean_similarity(
            _layer_A(layers=(4,)), self._timestamps(), _sections()
        )
        assert len(plt.get_fignums()) > before
        plt.close("all")

    def test_transform_tag_in_title(self, tmp_path):
        out = tmp_path / "lm_tag.png"
        plot_layer_mean_similarity(
            _layer_A(), self._timestamps(), _sections(),
            transform_tag="centered+whitened", output_path=out,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# _layer_mean_path
# ---------------------------------------------------------------------------

class TestLayerMeanPath:
    def test_inserts_layermean_suffix(self):
        assert _layer_mean_path("out.png") == Path("out_layermean.png")

    def test_preserves_extension(self):
        p = _layer_mean_path("results.npz")
        assert p.suffix == ".npz"
        assert "layermean" in p.stem

    def test_works_with_path_object(self):
        p = _layer_mean_path(Path("data/out.png"))
        assert p.name == "out_layermean.png"
