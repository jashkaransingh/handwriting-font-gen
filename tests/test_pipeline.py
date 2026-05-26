"""End-to-end smoke test, exercises the full pipeline at tiny scale."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_synthesize_and_load(tmp_path):
    from data.synthesize import generate_dataset
    from data.dataset import HandwritingDataset, load_classes

    out = generate_dataset(samples_per_class=5,
                           out_dir=str(tmp_path / "synth"),
                           augment_severity="light", seed=0)
    classes = load_classes(out)
    assert len(classes) == 62

    train_ds = HandwritingDataset(out / "train.npz")
    val_ds = HandwritingDataset(out / "val.npz")
    assert len(train_ds) + len(val_ds) == 62 * 5

    img, label = train_ds[0]
    assert img.shape == (1, 28, 28)
    assert 0 <= int(label) < 62


def test_inference_loads_checkpoint():
    """Only runs if a trained checkpoint exists."""
    ckpt_path = Path("checkpoints/best.pth")
    if not ckpt_path.exists():
        pytest.skip("no trained checkpoint")

    import torch
    from models.cnn import HandwritingCNN

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = HandwritingCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, len(classes))


def test_render_text_produces_image():
    """Only runs if dataset + checkpoint exist."""
    npz = Path("data/synthetic/train.npz")
    if not npz.exists():
        pytest.skip("no synthetic dataset, run data/synthesize.py first")

    from data.dataset import load_classes
    from inference.generate import CharacterBank, render_text

    classes = load_classes("data/synthetic")
    bank = CharacterBank(str(npz), classes, samples_per_class=20, seed=0)
    canvas = render_text("test", bank, seed=0)
    assert canvas.ndim == 2
    assert canvas.shape[0] > 0 and canvas.shape[1] > 0
    assert canvas.dtype == np.uint8
