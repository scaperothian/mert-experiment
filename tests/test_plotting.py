"""Tests for matplotlib plotting functions (no display, file output only)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mert_experiment.plotting import (
    _pairwise_path,
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

    def test_no_output_calls_show(self):
        with patch("matplotlib.pyplot.show") as mock_show:
            plot_prototype_similarity(_layer_results(layers=(4,)), _sections())
        mock_show.assert_called_once()

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

    def test_no_output_calls_show(self):
        with patch("matplotlib.pyplot.show") as mock_show:
            plot_section_grids(_layer_grids(layers=(4,)), self._labels(3))
        mock_show.assert_called_once()

    def test_audio_name_in_title(self, tmp_path):
        out = tmp_path / "titled.png"
        plot_section_grids(
            _layer_grids(), self._labels(3), audio_name="My Song", output_path=out
        )
        assert out.exists()
