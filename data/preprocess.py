import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def preprocess_sheet(image_path: str, output_dir: str, min_area: int = 200):
    """
    Takes a scan of handwriting on a white page.
    1. Converts to grayscale, applies adaptive threshold
    2. Finds contours (each contour = one character candidate)
    3. Filters noise by area, pads, resizes to 64x64
    4. Saves each character as a PNG
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold handles uneven lighting from phone scans
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=10
    )

    # Morphological closing to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue  # noise

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h
        if aspect > 5 or aspect < 0.1:
            continue  # probably a line artifact, not a character

        # Crop with padding
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        char_img = gray[y1:y2, x1:x2]
        char_img = cv2.resize(char_img, (64, 64), interpolation=cv2.INTER_AREA)
        char_img = cv2.bitwise_not(char_img)  # white on black

        out_path = os.path.join(output_dir, f"char_{i:04d}.png")
        cv2.imwrite(out_path, char_img)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of raw handwriting scans")
    parser.add_argument("--output", required=True, help="Output directory for character images")
    parser.add_argument("--min-area", type=int, default=200)
    args = parser.parse_args()

    total = 0
    for img_path in Path(args.input).glob("*.{png,jpg,jpeg}"):
        n = preprocess_sheet(str(img_path), args.output, args.min_area)
        print(f"{img_path.name}: {n} characters extracted")
        total += n

    print(f"\nTotal: {total} character images saved to {args.output}")


if __name__ == "__main__":
    main()
