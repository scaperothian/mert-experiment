"""Tests for JSON loading and audio ingestion (no real model required)."""

import json
import struct
import wave
from pathlib import Path

import pytest
import torch

from mert_experiment.io import load_audio, load_song
from mert_experiment.config import TARGET_SR


# ---- fixtures ---------------------------------------------------------------

def _write_song_json(path: Path, audio_path: str, groups: list) -> None:
    data = {
        "presentation": {
            "id": {"audio": audio_path},
            "groups": groups,
        }
    }
    path.write_text(json.dumps(data))


def _minimal_groups():
    return [
        {
            "slides": [
                {"text": "Verse 1\nSecond line", "start time": "0.0", "stop time": "10.0"},
                {"text": "Chorus\nLine 2",       "start time": "10.0", "stop time": "20.0"},
            ]
        },
        {
            "slides": [
                {"text": "", "start time": "20.0", "stop time": "30.0"},  # empty label
            ]
        },
    ]


def _write_wav(path: Path, sr: int = TARGET_SR, duration_sec: float = 0.5, channels: int = 1):
    """Write a silent WAV file for testing."""
    n_samples = int(sr * duration_sec)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n_samples * channels)


# ---- load_song --------------------------------------------------------------

class TestLoadSong:
    def test_returns_audio_path_and_sections(self, tmp_path):
        audio = tmp_path / "song.wav"
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, str(audio), _minimal_groups())

        audio_path, sections = load_song(json_file)
        assert isinstance(audio_path, Path)
        assert len(sections) == 3

    def test_section_fields_present(self, tmp_path):
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", _minimal_groups())

        _, sections = load_song(json_file)
        for sec in sections:
            assert "label" in sec
            assert "start" in sec
            assert "stop" in sec

    def test_label_is_first_line_only(self, tmp_path):
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", _minimal_groups())

        _, sections = load_song(json_file)
        assert sections[0]["label"] == "Verse 1"   # not "Verse 1\nSecond line"

    def test_label_truncated_to_40_chars(self, tmp_path):
        long_text = "A" * 60
        groups = [{"slides": [{"text": long_text, "start time": "0.0", "stop time": "5.0"}]}]
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", groups)

        _, sections = load_song(json_file)
        assert len(sections[0]["label"]) <= 40

    def test_empty_text_becomes_unlabeled(self, tmp_path):
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", _minimal_groups())

        _, sections = load_song(json_file)
        assert sections[2]["label"] == "(unlabeled)"

    def test_timestamps_are_floats(self, tmp_path):
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", _minimal_groups())

        _, sections = load_song(json_file)
        for sec in sections:
            assert isinstance(sec["start"], float)
            assert isinstance(sec["stop"], float)

    def test_multiple_groups_flattened(self, tmp_path):
        groups = [
            {"slides": [{"text": "A", "start time": "0", "stop time": "5"}]},
            {"slides": [{"text": "B", "start time": "5", "stop time": "10"},
                        {"text": "C", "start time": "10", "stop time": "15"}]},
        ]
        json_file = tmp_path / "song.json"
        _write_song_json(json_file, "audio.wav", groups)

        _, sections = load_song(json_file)
        assert len(sections) == 3

    def test_missing_key_raises(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text(json.dumps({"presentation": {"id": {}}}))  # no "audio" key
        with pytest.raises(KeyError):
            load_song(bad_json)


# ---- load_audio -------------------------------------------------------------

class TestLoadAudio:
    def test_mono_wav_returns_1d_tensor(self, tmp_path):
        wav_path = tmp_path / "mono.wav"
        _write_wav(wav_path, sr=TARGET_SR, channels=1)
        result = load_audio(wav_path)
        assert result.dim() == 1

    def test_stereo_wav_downmixed_to_mono(self, tmp_path):
        wav_path = tmp_path / "stereo.wav"
        _write_wav(wav_path, sr=TARGET_SR, channels=2)
        result = load_audio(wav_path)
        assert result.dim() == 1

    def test_sample_count_matches_duration(self, tmp_path):
        duration = 0.5
        wav_path = tmp_path / "half.wav"
        _write_wav(wav_path, sr=TARGET_SR, duration_sec=duration, channels=1)
        result = load_audio(wav_path)
        expected = int(TARGET_SR * duration)
        # allow ±1 sample for rounding
        assert abs(result.shape[0] - expected) <= 1

    def test_resampling_from_different_sr(self, tmp_path):
        # write at 16 kHz, expect resampled to TARGET_SR
        src_sr = 16_000
        duration = 0.25
        wav_path = tmp_path / "16k.wav"
        _write_wav(wav_path, sr=src_sr, duration_sec=duration, channels=1)
        result = load_audio(wav_path)
        expected = int(TARGET_SR * duration)
        # resampling introduces small rounding; allow 5% tolerance
        assert abs(result.shape[0] - expected) < expected * 0.05

    def test_output_is_float_tensor(self, tmp_path):
        wav_path = tmp_path / "f.wav"
        _write_wav(wav_path, sr=TARGET_SR, channels=1)
        result = load_audio(wav_path)
        assert result.dtype == torch.float32
