"""MERT model loading and full-song inference."""

import math

import torch

from .config import MODEL_ID, TARGET_SR

EMBED_CHUNK_SEC = 30.0  # audio processed per forward pass


def load_model(device: str | None = None):
    """
    Download (or load from cache) MERT and its feature extractor.

    Returns:
        processor, model  (model already on device, in eval mode)
    """
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = (
        AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        .to(device)
        .eval()
    )
    return processor, model


def embed_full_song(
    waveform: torch.Tensor,
    model,
    processor,
    device: str,
    chunk_sec: float = EMBED_CHUNK_SEC,
) -> torch.Tensor:
    """
    Run MERT over an entire waveform in fixed-size chunks and concatenate.

    Chunking keeps memory bounded on long songs and enables a progress bar.
    Each chunk is processed independently; hidden states are concatenated along
    the time axis, so the output is identical in shape to a single-pass run.

    Args:
        waveform:  1-D float tensor at TARGET_SR
        model:     MERT AutoModel (on device, eval mode)
        processor: Wav2Vec2FeatureExtractor
        chunk_sec: seconds of audio per forward pass (default 30 s)

    Returns:
        hidden states tensor of shape [num_layers + 1, num_frames, hidden_dim]
    """
    from tqdm import tqdm

    chunk_samples = int(chunk_sec * TARGET_SR)
    total_samples = waveform.shape[0]
    n_chunks = math.ceil(total_samples / chunk_samples)
    duration_sec = total_samples / TARGET_SR

    all_hidden: list[torch.Tensor] = []
    bar = tqdm(
        total=duration_sec,
        unit="s",
        unit_scale=False,
        desc="Embedding",
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
    )
    with bar:
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[start:end]

            inputs = processor(
                chunk.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            # hidden_states: tuple of (L+1) tensors, each [1, T, D]
            hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)
            all_hidden.append(hidden.cpu())
            bar.update((end - start) / TARGET_SR)

    return torch.cat(all_hidden, dim=1)  # [L+1, total_T, D]
