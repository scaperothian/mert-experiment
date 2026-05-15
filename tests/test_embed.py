"""Tests for MERT inference layer — model is mocked, no download required."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from mert_experiment.embed import embed_full_song
from mert_experiment.config import TARGET_SR


def _make_mock_model(n_layers: int = 13, n_frames: int = 200, hidden_dim: int = 768):
    """Return a fake MERT model whose output mimics the real one's structure."""
    hidden_states = tuple(
        torch.randn(1, n_frames, hidden_dim) for _ in range(n_layers)
    )
    out = SimpleNamespace(hidden_states=hidden_states)
    model = MagicMock()
    model.return_value = out
    return model, n_layers, n_frames, hidden_dim


def _make_mock_processor():
    """Return a fake processor that echoes its inputs back as a dict."""
    processor = MagicMock()
    processor.return_value = MagicMock()
    # .to(device) should return the same mock
    processor.return_value.to.return_value = processor.return_value
    return processor


class TestEmbedFullSong:
    def _run(self, n_layers=13, n_frames=200, hidden_dim=768, duration_sec=2.0):
        model, n_layers, n_frames, hidden_dim = _make_mock_model(n_layers, n_frames, hidden_dim)
        processor = _make_mock_processor()
        waveform = torch.zeros(int(TARGET_SR * duration_sec))

        with patch("torch.no_grad"):
            hidden = embed_full_song(waveform, model, processor, device="cpu")
        return hidden, n_layers, n_frames, hidden_dim

    def test_output_shape(self):
        hidden, n_layers, n_frames, hidden_dim = self._run()
        assert hidden.shape == (n_layers, n_frames, hidden_dim)

    def test_output_is_cpu_tensor(self):
        hidden, *_ = self._run()
        assert hidden.device.type == "cpu"

    def test_output_is_float32(self):
        hidden, *_ = self._run()
        assert hidden.dtype == torch.float32

    def test_layer_count_preserved(self):
        for n in [5, 10, 14]:
            hidden, n_layers, *_ = self._run(n_layers=n)
            assert hidden.shape[0] == n

    def test_processor_called_with_waveform(self):
        model, *_ = _make_mock_model()
        processor = _make_mock_processor()
        waveform = torch.ones(TARGET_SR)

        with patch("torch.no_grad"):
            embed_full_song(waveform, model, processor, device="cpu")

        call_args = processor.call_args
        passed_audio = call_args[0][0]
        assert torch.allclose(torch.tensor(passed_audio), waveform)

    def test_processor_receives_correct_sample_rate(self):
        model, *_ = _make_mock_model()
        processor = _make_mock_processor()
        waveform = torch.zeros(TARGET_SR)

        with patch("torch.no_grad"):
            embed_full_song(waveform, model, processor, device="cpu")

        kwargs = processor.call_args[1]
        assert kwargs["sampling_rate"] == TARGET_SR
