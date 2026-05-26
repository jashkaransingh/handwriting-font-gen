"""
Synthetic handwriting-style dataset generator.

When real handwriting samples are not available, this script generates a
training dataset by rendering characters in a variety of system fonts and then
running them through a heavy augmentation pipeline that simulates the kind of
variation a human hand produces.

The resulting dataset is balanced across 62 classes (digits + lowercase +
uppercase letters) and each sample is a 28x28 grayscale image, the same shape
that EMNIST uses, so the rest of the pipeline is dataset-agnostic.

If you have your own labeled handwriting samples, use data/label_gui.py instead
and skip this script.
"""

import argparse
import os
import string
import subprocess
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .augment import AugmentationPipeline


CLASSES = list(string.digits) + list(string.ascii_lowercase) + list(string.ascii_uppercase)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 28


def list_system_fonts(max_fonts: int = 30):
    """Find usable TrueType fonts on the system."""
    try:
        out = subprocess.check_output(["fc-list", ":lang=en"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return _fallback_fonts()

    paths = []
    for line in out.splitlines():
        path = line.split(":")[0].strip()
        if not path or not path.lower().endswith((".ttf", ".otf")):
            continue
        # Filter out CJK and symbol fonts that do not render latin characters well
        lower = path.lower()
        if any(skip in lower for skip in ["cjk", "noto-cjk", "symbol", "emoji",
                                           "math", "wingding", "kacst",
                                           "lohit", "padauk", "tlwg",
                                           "telugu", "tamil", "bengali",
                                           "devanagari", "khmer", "lao",
                                           "thai", "tibetan"]):
            continue
        paths.append(path)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if not unique:
        return _fallback_fonts()
    return unique[:max_fonts]


def _fallback_fonts():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    return [p for p in candidates if os.path.exists(p)]


def render_character(char: str, font_path: str, size: int = 28,
                     point_size: int = 22, offset_jitter: int = 2,
                     rng=None) -> np.ndarray:
    """Render a single character to a 28x28 grayscale numpy array."""
    rng = rng or np.random.default_rng()
    try:
        font = ImageFont.truetype(font_path, point_size)
    except OSError:
        font = ImageFont.load_default()

    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (size - text_w) // 2 - bbox[0] + int(rng.integers(-offset_jitter, offset_jitter + 1))
    y = (size - text_h) // 2 - bbox[1] + int(rng.integers(-offset_jitter, offset_jitter + 1))

    draw.text((x, y), char, fill=255, font=font)
    return np.array(img, dtype=np.uint8)


def generate_dataset(samples_per_class: int = 400, out_dir: str = "data/synthetic",
                     augment_severity: str = "medium", seed: int = 42):
    """
    Build a synthetic dataset of (samples_per_class * 62) images.

    Each character gets rendered in every available font multiple times. Each
    render goes through the augmentation pipeline. This is where the data
    pipeline does its actual job, the CNN sees the same character in dozens of
    plausible variations rather than dozens of pixel-identical copies.
    """
    rng = np.random.default_rng(seed)
    fonts = list_system_fonts(max_fonts=24)
    print(f"using {len(fonts)} fonts")
    pipeline = AugmentationPipeline(severity=augment_severity, seed=seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    for char in tqdm(CLASSES, desc="generating"):
        label_idx = CLASS_TO_IDX[char]
        for i in range(samples_per_class):
            font_path = fonts[i % len(fonts)]
            point_size = int(rng.integers(18, 26))
            base = render_character(char, font_path, IMG_SIZE, point_size, rng=rng)
            aug = pipeline(base)
            images.append(aug)
            labels.append(label_idx)

    images = np.stack(images).astype(np.uint8)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle so batches mix classes
    perm = rng.permutation(len(images))
    images = images[perm]
    labels = labels[perm]

    # Train and validation split, 90/10
    split = int(0.9 * len(images))
    np.savez_compressed(out_path / "train.npz",
                        images=images[:split], labels=labels[:split])
    np.savez_compressed(out_path / "val.npz",
                        images=images[split:], labels=labels[split:])

    with open(out_path / "classes.txt", "w") as f:
        for c in CLASSES:
            f.write(c + "\n")

    print(f"wrote {len(images[:split])} train, {len(images[split:])} val samples to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=400,
                        help="samples per class")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--severity", default="medium",
                        choices=["light", "medium", "heavy"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.samples, args.out, args.severity, args.seed)
