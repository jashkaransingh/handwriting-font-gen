"""
Data augmentation for handwriting character images.

The point of this module is to teach the CNN that the same character can look
many different ways depending on how a person actually writes. Real handwriting
has slant, varying stroke widths, ink density changes, slight rotations, and
noise from the scanning process. We simulate all of that here.

Speed matters because augmentation happens on every training batch. Earlier
iterations of this pipeline ran each transform serially on a single sample
which made training prep take hours. Everything here is now vectorized over
NumPy arrays and runs in milliseconds per sample.
"""

import numpy as np
import cv2
from typing import Optional


def random_rotate(img: np.ndarray, max_angle: float = 15.0,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Random rotation within +/- max_angle degrees, preserves image size."""
    rng = rng or np.random.default_rng()
    angle = rng.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=0,
                          flags=cv2.INTER_LINEAR)


def random_shift(img: np.ndarray, max_pixels: int = 2,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Random pixel-level translation in x and y."""
    rng = rng or np.random.default_rng()
    dx = int(rng.integers(-max_pixels, max_pixels + 1))
    dy = int(rng.integers(-max_pixels, max_pixels + 1))
    h, w = img.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matrix, (w, h), borderValue=0)


def random_shear(img: np.ndarray, max_shear: float = 0.2,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Random horizontal shear, mimics slant variations in handwriting."""
    rng = rng or np.random.default_rng()
    shear = rng.uniform(-max_shear, max_shear)
    h, w = img.shape[:2]
    matrix = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
    return cv2.warpAffine(img, matrix, (w, h), borderValue=0)


def morphological(img: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Random morphological op (dilate or erode) to simulate varying stroke widths.
    Some people press hard (thick strokes), some write lightly (thin strokes).
    """
    rng = rng or np.random.default_rng()
    op = rng.choice(["dilate", "erode", "none"])
    if op == "none":
        return img
    kernel = np.ones((2, 2), np.uint8)
    if op == "dilate":
        return cv2.dilate(img, kernel, iterations=1)
    return cv2.erode(img, kernel, iterations=1)


def gaussian_noise(img: np.ndarray, std: float = 8.0,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Synthetic noise to simulate scanning artifacts."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, std, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def elastic_distortion(img: np.ndarray, alpha: float = 8.0, sigma: float = 4.0,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Elastic deformation. This is the most powerful augmentation for handwriting
    because it bends strokes locally the way a human hand actually does.
    Based on Simard et al 2003.
    """
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32),
                          (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)


def stroke_thickness(img: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Lightly thicken or thin strokes by adjusting threshold post-blur."""
    rng = rng or np.random.default_rng()
    if rng.random() < 0.5:
        return img
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    threshold = int(rng.integers(80, 160))
    _, out = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return out


class AugmentationPipeline:
    """
    Composable augmentation pipeline. Apply a randomized chain of transforms
    on the fly during training.

    The order matters. Geometric transforms first, then morphological, then
    pixel-level noise last. Otherwise noise gets warped and ends up looking
    structured instead of random.
    """

    def __init__(self, severity: str = "medium", seed: Optional[int] = None):
        self.severity = severity
        self.rng = np.random.default_rng(seed)

        if severity == "light":
            self.config = dict(rotate=8, shear=0.1, shift=1, noise=4,
                               elastic=False, morph_prob=0.3)
        elif severity == "medium":
            self.config = dict(rotate=12, shear=0.2, shift=2, noise=6,
                               elastic=True, morph_prob=0.5)
        else:  # heavy
            self.config = dict(rotate=18, shear=0.3, shift=3, noise=10,
                               elastic=True, morph_prob=0.7)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        cfg = self.config
        out = img.copy()

        out = random_rotate(out, cfg["rotate"], self.rng)
        out = random_shear(out, cfg["shear"], self.rng)
        out = random_shift(out, cfg["shift"], self.rng)

        if self.rng.random() < cfg["morph_prob"]:
            out = morphological(out, self.rng)

        if cfg["elastic"] and self.rng.random() < 0.5:
            out = elastic_distortion(out, alpha=6.0, sigma=3.0, rng=self.rng)

        out = gaussian_noise(out, cfg["noise"], self.rng)
        return out


if __name__ == "__main__":
    # Sanity check, generate one augmented sample for visual inspection.
    import matplotlib.pyplot as plt
    test_img = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(test_img, "A", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)

    pipe = AugmentationPipeline(severity="medium", seed=0)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax in axes.flat:
        ax.imshow(pipe(test_img), cmap="gray")
        ax.axis("off")
    plt.suptitle("Augmentation samples")
    plt.savefig("/tmp/aug_test.png")
    print("saved /tmp/aug_test.png")
