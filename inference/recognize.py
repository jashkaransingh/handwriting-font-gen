"""
Recognition CLI. Given an image (any size), classify it as one of the 62
characters the model knows.

This is the inverse of generate.py. Useful for sanity checking the model on
real handwriting samples and as a building block for downstream OCR tasks.
"""

import argparse

import numpy as np
import torch
from PIL import Image

from data.preprocess import normalize_glyph
from models.cnn import HandwritingCNN


def load_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = HandwritingCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, classes


def recognize(image_path: str, model, classes, device: str = "cpu",
              top_k: int = 3):
    img = np.array(Image.open(image_path).convert("L"))

    # If image already has black background (text rendered light on dark),
    # use it directly. Otherwise invert so ink is bright.
    if img.mean() > 127:
        img = 255 - img

    # Threshold then normalize to 28x28 centered
    _, binary = __import__("cv2").threshold(img, 50, 255, 0)
    glyph = normalize_glyph(binary, out_size=28)

    tensor = torch.from_numpy(glyph).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_p, top_idx = probs.topk(min(top_k, len(classes)))

    return [(classes[int(i)], float(p)) for p, i in zip(top_p, top_idx)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="checkpoints/best.pth")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model, classes = load_model(args.model)
    results = recognize(args.image, model, classes, top_k=args.top_k)
    for char, prob in results:
        print(f"  {char}  {prob:.3f}")


if __name__ == "__main__":
    main()
