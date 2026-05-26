import torch
import numpy as np
import cv2
import argparse
from models.model import HandwritingCNN

CHAR_WIDTH = 64
CHAR_HEIGHT = 64
LETTER_SPACING = 4
SPACE_WIDTH = 24


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    classes = ckpt["classes"]
    model = HandwritingCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes


def char_to_image(char, model, classes, device):
    """Use the model to generate the character image (identity lookup from training)."""
    if char not in classes:
        return None
    idx = classes.index(char)
    # In a real system, we'd use a decoder or retrieve the prototype.
    # Here we return a blank (this method works with labeled prototype retrieval).
    return np.zeros((CHAR_HEIGHT, CHAR_WIDTH), dtype=np.uint8)


def render_text(text, model, classes, prototype_dir=None):
    """
    Render text string as an image using prototype character images.
    """
    chars = []
    for char in text:
        if char == " ":
            chars.append(("space", None))
        elif char.lower() in [c.lower() for c in classes]:
            # Load prototype from labeled dataset
            chars.append((char, char))
        else:
            chars.append((char, None))

    # Build canvas
    total_width = sum(
        SPACE_WIDTH if c[0] == "space" else CHAR_WIDTH + LETTER_SPACING
        for c in chars
    )
    canvas = np.ones((CHAR_HEIGHT + 20, total_width + 20), dtype=np.uint8) * 255

    x = 10
    for char, label in chars:
        if char == "space":
            x += SPACE_WIDTH
        elif label:
            # In practice: load prototype image from prototype_dir/label/*.png
            # Placeholder: draw char name as text for demo
            cv2.putText(canvas, char, (x, CHAR_HEIGHT),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
            x += CHAR_WIDTH + LETTER_SPACING

    return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.png")
    args = parser.parse_args()

    model, classes = load_model(args.model)
    img = render_text(args.text, model, classes)
    cv2.imwrite(args.output, img)
    print(f"Saved to {args.output}")
