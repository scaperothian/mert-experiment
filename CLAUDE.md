# CLAUDE.md — MERT Experiment

## What this project does

Explores MERT audio embeddings for the **live slide-advancement** use case:
given a clean recording with known section boundaries, build a pooled prototype
embedding per section, then compare every windowed frame of the audio against
those prototypes. The section whose prototype scores highest at a given moment
is the predicted current section.

MERT has 12 transformer layers. Layers 10–12 capture musical semantics (melody,
chord function, section type). Layers 4–6 capture low-level acoustic/timbral
features. Probing multiple layers shows which level of representation is most
discriminative for a given recording.

## Commands to know

```bash
# Run all tests (fast — no model download)
.venv/bin/pytest
.venv/bin/pytest --cov=mert_experiment --cov-report=term-missing

# Analyse a song (plot shown by default)
.venv/bin/python -m mert_experiment.cli song.json
.venv/bin/python -m mert_experiment.cli song.json --plot-output sim.png

# Save everything to disk
.venv/bin/python -m mert_experiment.cli song.json --save-npz results.npz

# Also compute and plot section×section similarity grid
.venv/bin/python -m mert_experiment.cli song.json --also-pairwise
.venv/bin/python -m mert_experiment.cli song.json --also-pairwise \
    --plot-output sim.png       # saves sim.png + sim_pairwise.png
.venv/bin/python -m mert_experiment.cli song.json --also-pairwise \
    --save-npz out.npz          # saves out.npz + out_pairwise.npz

# Audio file with named sections
.venv/bin/python -m mert_experiment.cli song.wav \
  --section Intro 0 30 --section Verse 30 90 --section Chorus 90 130

# Suppress plot
.venv/bin/python -m mert_experiment.cli song.json --no-plot
```

## Package layout

| Module | Key exports |
|---|---|
| `config.py` | `MODEL_ID`, `TARGET_SR`, `WINDOW_SEC`, `HOP_SEC`, `LAYERS_TO_PROBE`, `MERT_FRAME_RATE` |
| `io.py` | `load_song` (JSON → sections list), `load_audio` (→ mono 24 kHz tensor) |
| `embed.py` | `load_model`, `embed_full_song` → `[L+1, T, D]` hidden states |
| `windows.py` | `section_windows`, `whole_song_window_spans`, `window_embeddings`, `pool_normalized`, `pool_section`, `frame_to_section_assignment` |
| `similarity.py` | `cosine_matrix` (square), `cosine_block` (rectangular M×N), `print_matrix` |
| `plotting.py` | `plot_prototype_similarity` (line chart), `plot_section_grids` (heatmap grid), `_pairwise_path` |
| `cli.py` | `mert-sim` entry point, `analyze_layer`, `_run` |

## Core data flow

```
JSON / audio file
  → load_song / load_audio
  → embed_full_song              [L+1, T, D] hidden states

per layer:
  section_windows + window_embeddings + pool_normalized
    → P  [N_sec, D]              section prototypes
  whole_song_window_spans + window_embeddings
    → F  [N_frames, D]           frame embeddings
  cosine_block(P, F)
    → A  [N_sec, N_frames]       prototype-vs-frame similarity  ← VERSION A

plot_prototype_similarity(A, timestamps, sections)
    → line plot, one subplot per layer, shared time axis
    → vertical dashed lines at section boundaries

--also-pairwise:
  cosine_matrix(P)
    → [N_sec, N_sec]             pooled section×section
  plot_section_grids(...)
    → heatmap grid with numeric scores
```

## Test strategy

| File | What it covers |
|---|---|
| `test_windows.py` | `section_windows`, `whole_song_window_spans`, `window_embeddings`, `pool_normalized`, `pool_section`, `frame_to_section_assignment` |
| `test_similarity.py` | `cosine_matrix`, `cosine_block`, `print_matrix` (incl. `mark_columns`) |
| `test_io.py` | JSON parsing, WAV loading, resampling, stereo→mono |
| `test_embed.py` | `embed_full_song` with mocked model — no download |
| `test_plotting.py` | `plot_prototype_similarity`, `plot_section_grids`, `_pairwise_path` — file output |

All 99 tests run in ~3 seconds with no internet access.

## Output files

| File | Contents |
|---|---|
| `results.npz` | `timestamps`, `section_starts/stops`, `layer{N}_A/P/F` per layer |
| `results_pairwise.npz` | `layer{N}` → `[N_sec, N_sec]` pooled similarity per layer |
| `sim.png` | Prototype-vs-frame line plot |
| `sim_pairwise.png` | Section×section grid heatmap |

## Adding experiments

1. Change `LAYERS_TO_PROBE` in `config.py` to target specific layers.
2. Experiment with `pool_normalized` in `windows.py` — try max-pool or
   attention-weighted mean instead of simple mean.
3. Run with `--save-npz` and load arrays in a notebook for deeper analysis.
4. Add a test in `tests/` to lock in any invariant you discover.

## Dependencies

- `torch` + `torchaudio` — must match minor version (both 2.7.x)
- `transformers` — `AutoModel` + `Wav2Vec2FeatureExtractor` for MERT
- `soundfile` — torchaudio backend for WAV/FLAC without ffmpeg
- `numpy` — matrix arithmetic
- `matplotlib` — line plot and section grid visualisations
