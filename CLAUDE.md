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

# Analyse a song — centering + ZCA whitening ON by default
.venv/bin/python -m mert_experiment.cli song.json
.venv/bin/python -m mert_experiment.cli song.json --plot-output sim.png
# saves sim.png (per-layer) + sim_layermean.png (layer-mean + smoothed)

# Disable transforms selectively
.venv/bin/python -m mert_experiment.cli song.json --no-whiten          # center only
.venv/bin/python -m mert_experiment.cli song.json --no-center --no-whiten  # raw baseline

# Windowing: 1-second look-back updated every 100 ms
.venv/bin/python -m mert_experiment.cli song.json --window 1.0 --hop 0.1

# Smoothing: layer-mean plot averaged over 5 samples instead of 3
.venv/bin/python -m mert_experiment.cli song.json --smooth-k 5

# Save everything to disk
.venv/bin/python -m mert_experiment.cli song.json --save-npz results.npz

# Also compute and plot section×section similarity grid
.venv/bin/python -m mert_experiment.cli song.json --also-pairwise
.venv/bin/python -m mert_experiment.cli song.json --also-pairwise \
    --plot-output sim.png       # saves sim.png + sim_layermean.png + sim_pairwise.png
.venv/bin/python -m mert_experiment.cli song.json --also-pairwise \
    --save-npz out.npz          # saves out.npz + out_pairwise.npz

# Audio file — treated as one section covering the full duration
.venv/bin/python -m mert_experiment.cli song.wav

# Suppress plot
.venv/bin/python -m mert_experiment.cli song.json --no-plot
```

## Package layout

| Module | Key exports |
|---|---|
| `config.py` | `MODEL_ID`, `TARGET_SR`, `WINDOW_SEC`, `HOP_SEC`, `LAYERS_TO_PROBE`, `MERT_FRAME_RATE` |
| `io.py` | `load_song` (JSON → sections list), `load_audio` (→ mono 24 kHz tensor) |
| `embed.py` | `load_model`, `embed_full_song` → `[L+1, T, D]` hidden states |
| `windows.py` | `section_windows`, `whole_song_window_spans`, `window_means` (raw), `window_embeddings` (L2-normalized), `pool_normalized`, `pool_section`, `frame_to_section_assignment` |
| `transform.py` | `fit_mu`, `fit_whitening_matrix`, `apply_transform`, `l2_normalize_rows`, `fit_transform` |
| `similarity.py` | `cosine_matrix` (square), `cosine_block` (rectangular M×N), `print_matrix` |
| `plotting.py` | `plot_prototype_similarity` (per-layer line chart), `plot_layer_mean_similarity` (layer-mean + smoothed), `plot_section_grids` (heatmap grid), `causal_rolling_mean`, `_pairwise_path`, `_layer_mean_path` |
| `cli.py` | `mert-sim` entry point, `analyze_layer`, `_run` |

## Core data flow

```
JSON / audio file
  → load_song / load_audio
  → embed_full_song              [L+1, T, D] hidden states

per layer:
  whole_song_window_spans + window_means
    → F_raw  [N_frames, D]       raw un-normalized window means

  fit_transform(F_raw, center=True, whiten=True)
    → mu [1, D], W [D, D]        centering mean + ZCA whitening matrix

  l2_normalize_rows(apply_transform(F_raw, mu, W))
    → F  [N_frames, D]           centered + whitened frame embeddings

  per section:
    section_windows + window_means → raw_wins
    apply_transform(raw_wins, mu, W) → mean(dim=0) → l2_normalize
    → P  [N_sec, D]              section prototypes (transform-before-average!)

  cosine_block(P, F)
    → A  [N_sec, N_frames]       prototype-vs-frame similarity  ← VERSION A

  argmax(A) vs ground truth labels → accuracy
  mean off-diagonal cosine_block(P, P) → separation metric

plot_prototype_similarity(A, timestamps, sections)
    → per-layer line plot, one subplot per layer, shared time axis
    → vertical dashed lines at section boundaries

plot_layer_mean_similarity(layer_A, timestamps, sections)
    → two-panel summary: mean across layers + causal rolling-mean smoothed version
    → smooth_k controls the rolling-mean window (default 3, CLI: --smooth-k)
    → y-axis auto-scales to data range in both plots

Both plot functions leave figures open; _run calls plt.show() once after all
figures are built so they appear together in interactive mode.

--also-pairwise:
  cosine_matrix(P)
    → [N_sec, N_sec]             pooled section×section
  plot_section_grids(...)
    → heatmap grid with numeric scores
```

**Critical ordering:** `apply_transform` → `mean` → `l2_normalize` for prototypes.
Averaging raw windows then normalizing reintroduces the song-mean that was removed by centering.

## Test strategy

| File | What it covers |
|---|---|
| `test_windows.py` | `section_windows`, `whole_song_window_spans`, `window_means`, `window_embeddings`, `pool_normalized`, `pool_section`, `frame_to_section_assignment` |
| `test_transform.py` | `fit_mu`, `fit_whitening_matrix`, `apply_transform`, `l2_normalize_rows`, `fit_transform` |
| `test_similarity.py` | `cosine_matrix`, `cosine_block`, `print_matrix` (incl. `mark_columns`) |
| `test_io.py` | JSON parsing, WAV loading, resampling, stereo→mono |
| `test_embed.py` | `embed_full_song` with mocked model — no download |
| `test_plotting.py` | `plot_prototype_similarity`, `plot_layer_mean_similarity`, `plot_section_grids`, `causal_rolling_mean`, `_pairwise_path`, `_layer_mean_path` |

All 148 tests run in ~6 seconds with no internet access.

## Output files

| File | Contents |
|---|---|
| `results.npz` | `timestamps`, `section_starts/stops`, `transform`, `layer{N}_A/P/F` per layer |
| `results_pairwise.npz` | `layer{N}` → `[N_sec, N_sec]` pooled similarity per layer |
| `sim.png` | Per-layer prototype-vs-frame line plot (one subplot per layer) |
| `sim_layermean.png` | Layer-mean summary: raw mean + causal rolling-mean smoothed |
| `sim_pairwise.png` | Section×section grid heatmap (requires `--also-pairwise`) |

## Adding experiments

1. Change `LAYERS_TO_PROBE` in `config.py` to target specific layers.
2. Experiment with `pool_normalized` in `windows.py` — try max-pool or
   attention-weighted mean instead of simple mean.
3. Run with `--save-npz` and load arrays in a notebook for deeper analysis.
4. Compare transforms: run `--no-center --no-whiten` vs `--no-whiten` vs default
   and observe how argmax accuracy and off-diagonal prototype similarity change.
5. Try different windowing: `--window 1.0 --hop 0.1` gives a new embedding every
   100 ms while still looking back 1 second of audio. Smaller hop → more
   temporal resolution in the plots.
6. Tune smoothing: `--smooth-k 1` shows raw layer-mean scores; larger K smooths
   out frame-to-frame noise to reveal section-level trends.
7. Add a test in `tests/` to lock in any invariant you discover.

## Dependencies

- `torch` + `torchaudio` — must match minor version (both 2.7.x)
- `transformers` — `AutoModel` + `Wav2Vec2FeatureExtractor` for MERT
- `soundfile` — torchaudio backend for WAV/FLAC without ffmpeg
- `numpy` — matrix arithmetic
- `matplotlib` — line plot and section grid visualisations

## Why centering and whitening?

MERT (like most SSL models) suffers from **anisotropy**: embeddings cluster in
a narrow cone of the high-dimensional space, producing uniformly high cosine
similarities (0.85–0.95) regardless of musical content. Two unrelated sections
can score as similar as two identical ones.

**Centering** subtracts the song-mean vector — the "what music sounds like in
general" direction that dominates all embeddings. After centering, scores drop
and spread, but the space remains correlated.

**ZCA whitening** applies the inverse-square-root of the covariance matrix
(`transform.py: fit_whitening_matrix`). This decorrelates all dimensions and
equalises their variance. After both transforms, sections that sound alike score
≥ 0.7 and sections that differ score near 0 or negative.

The `transform.npz` key in `--save-npz` output records which transform was applied.
