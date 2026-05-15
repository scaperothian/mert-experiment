# MERT Experiment

Experiments with [MERT](https://huggingface.co/m-a-p/MERT-v1-95M) audio embeddings
for song section similarity and live slide-advancement research.

**Core idea:** build a pooled "prototype" embedding per section from a clean
recording, then compare every sliding-window frame of the audio against those
prototypes. The section with the highest cosine similarity at each moment tells
you where you are in the song — the same logic applies in live deployment with
streaming audio.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Note:** `torch` and `torchaudio` must be a matching pair (e.g. both 2.7.x).
> MERT weights (~380 MB) are downloaded automatically from HuggingFace on first
> run and cached in `~/.cache/huggingface/`.

## Running Tests

```bash
# All tests — no model download required (embed tests use mocks)
.venv/bin/pytest

# With coverage
.venv/bin/pytest --cov=mert_experiment --cov-report=term-missing

# One module
.venv/bin/pytest tests/test_windows.py -v
```

## `mert-sim`

For each probed MERT layer, computes a **[N\_sections × N\_frames]**
cosine-similarity matrix (prototypes vs. every windowed frame), reports argmax
accuracy, and by default shows a line-plot of similarity scores over time.

### From a ProPresenter JSON manifest

The JSON embeds the audio path and section boundaries:

```json
{
  "presentation": {
    "id": { "audio": "path/to/song.wav" },
    "groups": [
      {
        "slides": [
          { "text": "Verse 1\nLyric line", "start time": "0.0",  "stop time": "30.0" },
          { "text": "Chorus\nLine 2",      "start time": "30.0", "stop time": "60.0" }
        ]
      }
    ]
  }
}
```

```bash
mert-sim song.json
```

### From a raw audio file

```bash
# Full file as one section
mert-sim song.wav

# Named sections with timestamps (seconds)
mert-sim song.wav \
  --section Intro   0   30 \
  --section Verse1  30  90 \
  --section Chorus  90  130
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--no-plot` | off | Suppress all matplotlib output |
| `--plot-output FILE` | display | Save the line plot to FILE instead of showing it |
| `--also-pairwise` | off | Also compute and plot the section×section similarity grid |
| `--save-npz FILE` | off | Save VERSION A matrices and embeddings to a `.npz` file |

### Default behaviour — line plot

By default a line plot is shown after analysis. Each line is one section
prototype; the x-axis is time in seconds; vertical dashed bars mark section
boundaries from the JSON. One subplot per probed MERT layer, stacked vertically
with a shared time axis.

```bash
mert-sim song.json                          # display interactively
mert-sim song.json --plot-output sim.png   # save to file
mert-sim song.json --no-plot               # skip plot entirely
```

### Section×section grid (`--also-pairwise`)

Computes the pooled section×section cosine similarity and displays it as a
heatmap grid with numeric scores in each cell — one panel per layer.

If `--plot-output` is set, the grid is saved alongside with a `_pairwise`
suffix (e.g. `sim.png` → `sim_pairwise.png`).

If `--save-npz` is also set, the pairwise matrices are saved as a separate
`_pairwise.npz` file.

```bash
mert-sim song.json --also-pairwise
mert-sim song.json --also-pairwise --plot-output sim.png   # saves sim.png + sim_pairwise.png
mert-sim song.json --also-pairwise --save-npz out.npz      # saves out.npz + out_pairwise.npz
```

### Saving data for offline analysis

```bash
mert-sim song.json --save-npz results.npz
```

The `.npz` file contains:

| Key | Shape | Description |
|---|---|---|
| `section_starts` | `[N_sec]` | Section start times (s) |
| `section_stops` | `[N_sec]` | Section stop times (s) |
| `timestamps` | `[N_frames]` | Window centre times (s) |
| `layer{N}_A` | `[N_sec, N_frames]` | Prototype-vs-frame similarity |
| `layer{N}_P` | `[N_sec, D]` | Section prototype embeddings |
| `layer{N}_F` | `[N_frames, D]` | All-frame embeddings |

### Supported audio formats

`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

## Project Structure

```
mert_experiment/
  config.py    — model ID, sample rate, window/hop sizes, layers to probe
  io.py        — JSON manifest parsing, audio loading + resampling
  embed.py     — MERT model loading and full-song inference
  windows.py   — sliding-window spans, frame embedding, pooling, section assignment
  similarity.py — cosine_matrix, cosine_block, print_matrix
  plotting.py  — line plot (prototype similarity over time), section grid heatmap
  cli.py       — mert-sim entry point

tests/
  test_io.py        — JSON loading and audio ingestion
  test_windows.py   — all windowing and pooling functions
  test_similarity.py — cosine_matrix, cosine_block, print_matrix
  test_embed.py     — MERT inference (model mocked)
  test_plotting.py  — line plot and section grid (file output)
```

## Config Knobs (`mert_experiment/config.py`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `m-a-p/MERT-v1-95M` | HuggingFace model ID |
| `TARGET_SR` | `24000` | Required sample rate (Hz) |
| `WINDOW_SEC` | `1.0` | Sliding window length (s) |
| `HOP_SEC` | `0.5` | Hop between windows (s) |
| `LAYERS_TO_PROBE` | `[4,6,8,10,12]` | MERT layers to analyse |
| `MERT_FRAME_RATE` | `75` | MERT output frames per second |

Higher layers (10–12) capture musical semantics (melody, harmony, section
function). Lower layers (4–6) capture acoustic/timbral features.
