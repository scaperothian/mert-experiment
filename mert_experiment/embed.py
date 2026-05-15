"""MERT model loading and full-song inference."""

import torch

from .config import MODEL_ID, TARGET_SR


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
) -> torch.Tensor:
    """
    Run MERT once over an entire waveform.

    Args:
        waveform:  1-D float tensor at TARGET_SR
        model:     MERT AutoModel (on device, eval mode)
        processor: Wav2Vec2FeatureExtractor

    Returns:
        hidden states tensor of shape [num_layers + 1, num_frames, hidden_dim]
    """
    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (L+1) tensors, each [1, T, D]
    hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)  # [L+1, T, D]
    return hidden.cpu()
