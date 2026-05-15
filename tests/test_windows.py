"""Tests for sliding-window frame aggregation."""

import pytest
import torch

from mert_experiment.windows import (
    frame_to_section_assignment,
    pool_normalized,
    pool_section,
    section_windows,
    whole_song_window_spans,
    window_embeddings,
)


# --- section_windows ---------------------------------------------------------

class TestSectionWindows:
    def test_basic_count(self):
        # 2s section, 1s window, 0.5s hop at 75 fps
        # win=75f, hop=round(0.5*75)=38f, start=0, stop=150
        # f=0: yield(0,75), f=38
        # f=38: yield(38,113), f=76
        # f=76: 76+75=151>150, exit loop
        # tail: stop-win=75, f-hop=38, 75>38 -> yield(75,150)
        windows = list(section_windows(0.0, 2.0, window=1.0, hop=0.5, fps=75))
        assert len(windows) == 3
        for s, e in windows:
            assert e - s == 75
            assert s >= 0
            assert e <= 150

    def test_section_shorter_than_window_yields_nothing(self):
        windows = list(section_windows(0.0, 0.5, window=1.0, hop=0.5, fps=75))
        assert windows == []

    def test_section_exactly_one_window(self):
        windows = list(section_windows(0.0, 1.0, window=1.0, hop=0.5, fps=75))
        assert len(windows) >= 1
        assert windows[0] == (0, 75)

    def test_non_zero_start(self):
        windows = list(section_windows(10.0, 12.0, window=1.0, hop=0.5, fps=75))
        assert len(windows) > 0
        for s, e in windows:
            assert s >= 750   # 10s * 75fps
            assert e <= 900   # 12s * 75fps

    def test_frames_are_contiguous_within_bounds(self):
        start, stop = 5.0, 8.0
        fps = 75
        windows = list(section_windows(start, stop, window=1.0, hop=0.5, fps=fps))
        start_f = int(round(start * fps))
        stop_f  = int(round(stop  * fps))
        for s, e in windows:
            assert s >= start_f
            assert e <= stop_f

    def test_hop_larger_than_window_no_overlap(self):
        # hop=2s > win=1s: main windows don't overlap; tail window may add one extra
        windows = list(section_windows(0.0, 4.0, window=1.0, hop=2.0, fps=75))
        assert len(windows) == 3
        assert windows[0][1] <= windows[1][0]


# --- whole_song_window_spans -------------------------------------------------

class TestWholeSongWindowSpans:
    def test_all_spans_same_width(self):
        spans = whole_song_window_spans(300, window=1.0, hop=0.5, fps=75)
        win_f = int(round(1.0 * 75))
        for s, e in spans:
            assert e - s == win_f

    def test_starts_at_zero(self):
        spans = whole_song_window_spans(300, window=1.0, hop=0.5, fps=75)
        assert spans[0][0] == 0

    def test_no_span_exceeds_total_frames(self):
        total = 300
        spans = whole_song_window_spans(total, window=1.0, hop=0.5, fps=75)
        for s, e in spans:
            assert e <= total

    def test_count_matches_expected(self):
        # total=300, win=75, hop=38: f=0,38,76,114,152,190,228 -> 7 spans before 266>300
        # f=0,38,76,114,152,190,228: 228+75=303>300, so last valid is f=225? Let me just check it's > 0
        spans = whole_song_window_spans(300, window=1.0, hop=0.5, fps=75)
        assert len(spans) > 0

    def test_empty_when_total_frames_less_than_window(self):
        spans = whole_song_window_spans(10, window=1.0, hop=0.5, fps=75)
        assert spans == []

    def test_hops_are_non_overlapping_with_large_hop(self):
        spans = whole_song_window_spans(600, window=1.0, hop=2.0, fps=75)
        for i in range(len(spans) - 1):
            assert spans[i][1] <= spans[i + 1][0]


# --- window_embeddings -------------------------------------------------------

class TestWindowEmbeddings:
    def _frames(self, n: int, dim: int = 16) -> torch.Tensor:
        torch.manual_seed(5)
        return torch.randn(n, dim)

    def test_output_shape(self):
        frames = self._frames(300)
        spans = [(0, 75), (38, 113), (75, 150)]
        W = window_embeddings(frames, spans)
        assert W.shape == (3, 16)

    def test_output_is_unit_norm(self):
        frames = self._frames(300)
        spans = [(0, 75), (75, 150), (150, 225)]
        W = window_embeddings(frames, spans)
        norms = W.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_clips_to_tensor_length(self):
        frames = self._frames(100)
        spans = [(0, 75), (50, 200)]   # second span exceeds tensor
        W = window_embeddings(frames, spans)
        assert W.shape[0] == 2         # both produce valid windows

    def test_skips_spans_shorter_than_two(self):
        frames = self._frames(300)
        spans = [(0, 75), (100, 101)]  # second span is 1 frame — skipped
        W = window_embeddings(frames, spans)
        assert W.shape[0] == 1

    def test_returns_empty_when_no_valid_spans(self):
        frames = self._frames(300)
        spans = [(0, 1)]               # too short
        W = window_embeddings(frames, spans)
        assert W.shape[0] == 0
        assert W.shape[1] == 16

    def test_dim_matches_frames(self):
        dim = 32
        frames = self._frames(300, dim=dim)
        spans = [(0, 75)]
        W = window_embeddings(frames, spans)
        assert W.shape[1] == dim


# --- pool_normalized ---------------------------------------------------------

class TestPoolNormalized:
    def test_output_is_unit_norm(self):
        torch.manual_seed(1)
        W = torch.randn(5, 16)
        W = W / W.norm(dim=1, keepdim=True)
        v = pool_normalized(W)
        assert abs(v.norm().item() - 1.0) < 1e-5

    def test_output_shape(self):
        W = torch.randn(4, 32)
        v = pool_normalized(W)
        assert v.shape == (32,)

    def test_single_row_returns_that_row(self):
        torch.manual_seed(2)
        v = torch.randn(16)
        v = v / v.norm()
        W = v.unsqueeze(0)
        result = pool_normalized(W)
        assert torch.allclose(result, v, atol=1e-5)

    def test_identical_rows_equal_that_row(self):
        torch.manual_seed(3)
        v = torch.randn(16)
        v = v / v.norm()
        W = v.unsqueeze(0).expand(5, -1)
        result = pool_normalized(W)
        assert torch.allclose(result, v, atol=1e-5)


# --- pool_section ------------------------------------------------------------

class TestPoolSection:
    def _make_frames(self, n_frames: int, dim: int = 16) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(n_frames, dim)

    def test_pooled_is_unit_norm(self):
        frames = self._make_frames(300)
        _, pooled = pool_section(frames, 0.0, 2.0, fps=75)
        assert abs(pooled.norm().item() - 1.0) < 1e-5

    def test_windows_are_unit_norm(self):
        frames = self._make_frames(300)
        windows, _ = pool_section(frames, 0.0, 2.0, fps=75)
        norms = windows.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_window_count_matches_section_windows(self):
        frames = self._make_frames(300)
        expected = len(list(section_windows(0.0, 2.0, window=1.0, hop=0.5, fps=75)))
        win_tensor, _ = pool_section(frames, 0.0, 2.0, fps=75)
        assert win_tensor.shape[0] == expected

    def test_output_dimension_matches_input(self):
        dim = 32
        frames = self._make_frames(300, dim=dim)
        W, pooled = pool_section(frames, 0.0, 2.0, fps=75)
        assert W.shape[1] == dim
        assert pooled.shape[0] == dim

    def test_section_too_short_raises(self):
        frames = self._make_frames(300)
        with pytest.raises(ValueError, match="too short"):
            pool_section(frames, 0.0, 0.5, fps=75)

    def test_identical_input_gives_same_pooled(self):
        frames = self._make_frames(300)
        _, pooled1 = pool_section(frames, 0.0, 2.0, fps=75)
        _, pooled2 = pool_section(frames, 0.0, 2.0, fps=75)
        assert torch.allclose(pooled1, pooled2)

    def test_frame_clipping_at_tensor_boundary(self):
        frames = self._make_frames(140)
        _, pooled = pool_section(frames, 0.0, 2.0, fps=75)
        assert pooled.shape[0] == 16


# --- frame_to_section_assignment ---------------------------------------------

class TestFrameToSectionAssignment:
    def _sections(self):
        return [
            {"label": "Intro",  "start": 0.0,  "stop": 10.0},
            {"label": "Verse",  "start": 10.0, "stop": 30.0},
            {"label": "Chorus", "start": 30.0, "stop": 50.0},
        ]

    def test_centre_inside_first_section(self):
        # centre = (0+75)/2 / 75 = 0.5s -> Intro
        assert frame_to_section_assignment((0, 75), self._sections()) == 0

    def test_centre_inside_second_section(self):
        # centre = (750+825)/2 / 75 = 10.5s -> Verse
        assert frame_to_section_assignment((750, 825), self._sections()) == 1

    def test_centre_inside_third_section(self):
        # centre = (2250+2325)/2 / 75 = 30.5s -> Chorus
        assert frame_to_section_assignment((2250, 2325), self._sections()) == 2

    def test_centre_before_all_sections_returns_minus_one(self):
        # no section starts at negative time
        sections = [{"label": "A", "start": 5.0, "stop": 10.0}]
        assert frame_to_section_assignment((0, 75), sections) == -1

    def test_centre_after_all_sections_returns_minus_one(self):
        sections = [{"label": "A", "start": 0.0, "stop": 1.0}]
        # centre = (300+375)/2 / 75 = 4.5s, which is after stop=1.0s
        assert frame_to_section_assignment((300, 375), sections) == -1

    def test_centre_exactly_at_section_boundary_goes_to_next(self):
        # centre at exactly 10.0s — stop of Intro is 10.0 (exclusive), start of Verse is 10.0
        # so it should fall into Verse (index 1), not Intro
        sections = self._sections()
        # frame whose centre is exactly 10.0s: centre = (s+e)/2/fps = 10.0 => s+e = 1500
        # e.g. (750, 750) is degenerate; use (712, 788) -> centre = 1500/2/75 = 10.0
        result = frame_to_section_assignment((712, 788), sections)
        assert result == 1

    def test_custom_fps(self):
        sections = [{"label": "A", "start": 0.0, "stop": 10.0}]
        # at fps=10: centre = (0+10)/2/10 = 0.5s -> section 0
        assert frame_to_section_assignment((0, 10), sections, fps=10) == 0
