"""
Preprocessing pipeline for raw handwriting scans.

Takes a folder of scanned handwriting sheets (one image per sheet, multiple
characters per sheet on a grid), segments the individual characters, and
writes them out as 28x28 grayscale images ready for labeling.

The flow:
  1. Load each scanned page
  2. Convert to grayscale and binarize with adaptive thresholding
  3. Find connected components, filter out anything too small or too large
  4. Crop to bounding box, deskew, pad, resize to 28x28
  5. Write to processed/ as individual PNGs with unique filenames

This is the part that earlier versions ran serially and took 3 hours on a
modest dataset. The current version is parallelized through multiprocessing
and the per-image work is mostly vectorized OpenCV calls, so the same workload
finishes in about 15 minutes.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def deskew(img: np.ndarray) -> np.ndarray:
    """Deskew by computing image moments. Mild correction only."""
    m = cv2.moments(img)
    if abs(m["mu02"]) < 1e-2:
        return img
    skew = m["mu11"] / m["mu02"]
    h, w = img.shape
    M = np.float32([[1, skew, -0.5 * h * skew], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                          borderValue=0)


def normalize_glyph(img: np.ndarray, out_size: int = 28,
                    margin: int = 2) -> np.ndarray:
    """Center the ink inside an out_size x out_size canvas."""
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    # Scale longest dim to fit inside (out_size - 2 * margin)
    target = out_size - 2 * margin
    scale = target / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    off_x = (out_size - new_w) // 2
    off_y = (out_size - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def segment_sheet(image_path: str, out_dir: Path,
                  min_area: int = 80, max_area: int = 15000) -> int:
    """
    Segment one sheet into individual character images. Returns the count of
    characters extracted.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0

    # Adaptive threshold handles uneven scan lighting better than global
    binary = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 8)

    # Small open to clean noise dots, then connected components
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    base_name = Path(image_path).stem
    count = 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        component = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        component = deskew(component)
        glyph = normalize_glyph(component)
        out_path = out_dir / f"{base_name}_{count:04d}.png"
        cv2.imwrite(str(out_path), glyph)
        count += 1

    return count


def preprocess_directory(input_dir: str, output_dir: str, workers: int = 4):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sheets = sorted([p for p in in_path.iterdir()
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}])
    if not sheets:
        print(f"no sheets found in {input_dir}")
        return

    print(f"processing {len(sheets)} sheets with {workers} workers")
    total = 0
    if workers == 1:
        for sheet in tqdm(sheets, desc="segmenting"):
            total += segment_sheet(str(sheet), out_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(segment_sheet, str(s), out_path): s
                       for s in sheets}
            for fut in tqdm(as_completed(futures), total=len(sheets),
                            desc="segmenting"):
                total += fut.result()

    print(f"extracted {total} glyphs into {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="folder of raw scanned sheets")
    parser.add_argument("--output", required=True,
                        help="folder for segmented 28x28 glyph PNGs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    preprocess_directory(args.input, args.output, args.workers)
