import cv2
import numpy as np
import os
import random
from pathlib import Path
import argparse


def augment(img: np.ndarray, n: int = 8) -> list:
    """
    Generate n augmented versions of a character image.
    Augmentations chosen to mimic natural handwriting variation:
    slight rotation, elastic-ish distortion, stroke width variation.
    """
    augmented = []

    for _ in range(n):
        out = img.copy()

        # Random rotation (-12 to +12 degrees)
        angle = random.uniform(-12, 12)
        h, w = out.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderValue=0)

        # Random slight scale (0.9 to 1.1)
        scale = random.uniform(0.9, 1.1)
        out = cv2.resize(out, None, fx=scale, fy=scale)
        out = cv2.resize(out, (64, 64))

        # Random Gaussian noise
        noise = np.random.normal(0, 5, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Random dilation/erosion to vary stroke width
        kernel = np.ones((2, 2), np.uint8)
        if random.random() > 0.5:
            out = cv2.dilate(out, kernel, iterations=1)
        else:
            out = cv2.erode(out, kernel, iterations=1)

        augmented.append(out)

    return augmented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=8, help="augmentations per image")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    total = 0

    for img_path in Path(args.input).rglob("*.png"):
        label = img_path.parent.name  # expects labeled/A/char_0001.png structure
        label_dir = os.path.join(args.output, label)
        os.makedirs(label_dir, exist_ok=True)

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Save original
        cv2.imwrite(os.path.join(label_dir, img_path.name), img)

        # Save augmented versions
        for i, aug in enumerate(augment(img, args.n)):
            out_name = img_path.stem + f"_aug{i}.png"
            cv2.imwrite(os.path.join(label_dir, out_name), aug)

        total += args.n + 1

    print(f"Generated {total} images in {args.output}")


if __name__ == "__main__":
    main()
