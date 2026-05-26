"""
Inference layer for handwriting font generation.

Recognition is the easy half. The interesting half is turning a string of input
text into a rendered image that looks like the training style.

Approach:
  1. Build a per-class character bank by sampling high-confidence training
     images for each class
  2. For each character in the input text, retrieve a random sample from that
     class with a small random transformation so repeated chars do not look
     identical
  3. Crop each character to its actual ink bounds, normalize to a common
     baseline, then stitch horizontally with realistic spacing
  4. For spaces, insert a fixed-width gap

The baseline alignment step is what makes the output not look like ransom note
text. Descenders (g, j, p, q, y) hang below the baseline, ascenders (b, d, h,
k, l, t) reach above, and digits and capitals sit on different reference
points. Getting this right takes more care than the model itself.
"""

import argparse
import string
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from data.augment import AugmentationPipeline
from data.dataset import load_classes
from models.cnn import HandwritingCNN


# Standard typographic categories used for baseline calculation
DESCENDERS = set("gjpqy")
ASCENDERS = set("bdfhijklt")


class CharacterBank:
    """Holds a pool of training images per class for inference-time sampling."""

    def __init__(self, npz_path: str, classes, samples_per_class: int = 60,
                 seed: int = 0):
        data = np.load(npz_path)
        images = data["images"]
        labels = data["labels"]
        self.classes = classes
        rng = np.random.default_rng(seed)

        self.bank = {}
        for idx, char in enumerate(classes):
            mask = labels == idx
            class_imgs = images[mask]
            if len(class_imgs) > samples_per_class:
                pick = rng.choice(len(class_imgs), samples_per_class,
                                  replace=False)
                class_imgs = class_imgs[pick]
            self.bank[char] = class_imgs

    def sample(self, char: str, rng: np.random.Generator) -> Optional[np.ndarray]:
        if char not in self.bank or len(self.bank[char]) == 0:
            return None
        idx = rng.integers(0, len(self.bank[char]))
        return self.bank[char][idx].copy()


def crop_to_ink(img: np.ndarray, padding: int = 1) -> np.ndarray:
    """Crop to the bounding box of non-zero pixels, with optional padding."""
    if img.max() == 0:
        return img
    rows = np.any(img > 30, axis=1)
    cols = np.any(img > 30, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - padding)
    cmin = max(0, cmin - padding)
    rmax = min(img.shape[0], rmax + padding + 1)
    cmax = min(img.shape[1], cmax + padding + 1)
    return img[rmin:rmax, cmin:cmax]


def render_text(text: str, bank: CharacterBank,
                line_height: int = 56,
                baseline_ratio: float = 0.7,
                char_spacing: int = 6,
                space_width: int = 22,
                jitter_y: int = 2,
                seed: Optional[int] = None) -> np.ndarray:
    """
    Render input text using sampled characters from the bank.

    line_height controls total render height in pixels. baseline_ratio sets
    where the writing baseline sits within line_height (0.7 means 70% of the
    way down, which leaves room for descenders below).
    """
    rng = np.random.default_rng(seed)
    baseline_y = int(line_height * baseline_ratio)
    x_cap_height = int(line_height * 0.5)

    # Two-pass render. First pass collects each char's cropped image and
    # determines per-char vertical placement. Second pass composites.
    glyphs = []

    for char in text:
        if char == " ":
            glyphs.append(("space", space_width))
            continue

        sample = bank.sample(char, rng)
        if sample is None:
            # Try case-insensitive fallback
            for alt in (char.upper(), char.lower()):
                sample = bank.sample(alt, rng)
                if sample is not None:
                    break
        if sample is None:
            # Unknown char, insert a thin space
            glyphs.append(("space", space_width // 2))
            continue

        ink = crop_to_ink(sample, padding=1)
        if ink.size == 0:
            glyphs.append(("space", space_width // 2))
            continue

        # Scale ink to the writing height
        h, w = ink.shape
        target_h = x_cap_height if char.islower() and char not in ASCENDERS else int(line_height * 0.6)
        if char in DESCENDERS:
            target_h = int(line_height * 0.6)
        scale = target_h / max(h, 1)
        new_w = max(2, int(w * scale))
        new_h = max(2, int(h * scale))

        scaled = np.array(
            Image.fromarray(ink).resize((new_w, new_h), Image.BICUBIC))

        # Vertical placement relative to baseline
        if char in DESCENDERS:
            top = baseline_y - int(new_h * 0.75)
        elif char in ASCENDERS or char.isupper() or char.isdigit():
            top = baseline_y - new_h
        else:
            top = baseline_y - new_h

        top += int(rng.integers(-jitter_y, jitter_y + 1))
        glyphs.append(("char", scaled, top))

    # Compute final canvas width
    total_width = 0
    for g in glyphs:
        if g[0] == "space":
            total_width += g[1]
        else:
            total_width += g[1].shape[1] + char_spacing
    total_width = max(total_width, 1) + 10

    canvas = np.zeros((line_height, total_width), dtype=np.uint8)

    x = 5
    for g in glyphs:
        if g[0] == "space":
            x += g[1]
            continue
        scaled, top = g[1], g[2]
        h, w = scaled.shape
        x_end = min(x + w, canvas.shape[1])
        y_start = max(0, top)
        y_end = min(canvas.shape[0], top + h)
        sx_end = x_end - x
        sy_start = y_start - top
        sy_end = sy_start + (y_end - y_start)
        if sx_end > 0 and y_end > y_start:
            canvas[y_start:y_end, x:x_end] = np.maximum(
                canvas[y_start:y_end, x:x_end],
                scaled[sy_start:sy_end, :sx_end])
        x += w + char_spacing

    return canvas


def evaluate_recognition(model_path: str, val_npz: str, device: str = "cpu"):
    """Run the trained model on the validation set and report accuracy."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = HandwritingCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    data = np.load(val_npz)
    images = data["images"]
    labels = data["labels"]

    tensors = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    tensors = (tensors - 0.5) / 0.5

    correct = 0
    with torch.no_grad():
        for i in range(0, len(tensors), 256):
            batch = tensors[i:i + 256].to(device)
            out = model(batch)
            preds = out.argmax(dim=1).cpu().numpy()
            correct += (preds == labels[i:i + 256]).sum()

    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default="checkpoints/best.pth")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--out", default="output.png")
    parser.add_argument("--line-height", type=int, default=56)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    classes = load_classes(args.data_dir)
    train_npz = Path(args.data_dir) / "train.npz"
    bank = CharacterBank(str(train_npz), classes,
                         samples_per_class=80, seed=0)

    canvas = render_text(args.text, bank,
                         line_height=args.line_height,
                         seed=args.seed)

    # Invert to black-on-white for typical viewing
    out_img = 255 - canvas
    Image.fromarray(out_img).save(args.out)
    print(f"rendered {len(args.text)} chars to {args.out}")
    print(f"  canvas size: {canvas.shape}")


if __name__ == "__main__":
    main()
