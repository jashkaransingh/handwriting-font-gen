"""
Matplotlib-based labeling GUI for assigning character labels to segmented
handwriting glyphs.

Run this after data/preprocess.py to label each segmented glyph with its
character. The GUI shows you one glyph at a time, you press the key for the
character it represents, the label gets stored, and you advance. Press
backspace to undo. Press s to skip a glyph.

When finished, the labeled set gets written out to data/custom/train.npz and
data/custom/val.npz in the same format as data/synthesize.py produces, so the
rest of the pipeline (train, generate) works unchanged.

A small GUI tool like this exists because the original workflow of manually
typing labels into a spreadsheet alongside filenames was destroying me. This
takes the same input and reduces it to one keypress per glyph.
"""

import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image


VALID_CHARS = set(string.digits + string.ascii_letters)


def load_glyphs(folder: str):
    paths = sorted([p for p in Path(folder).iterdir()
                    if p.suffix.lower() == ".png"])
    if not paths:
        raise SystemExit(f"no PNGs found in {folder}")
    glyphs = []
    for p in paths:
        img = np.array(Image.open(p).convert("L"))
        glyphs.append((p.name, img))
    return glyphs


def label_session(folder: str, save_path: str, val_split: float = 0.1):
    glyphs = load_glyphs(folder)
    print(f"loaded {len(glyphs)} glyphs to label")

    labels_path = Path(save_path).with_suffix(".partial.json")
    if labels_path.exists():
        with open(labels_path) as f:
            saved = json.load(f)
        print(f"resuming from {labels_path}, {len(saved)} already labeled")
    else:
        saved = {}

    state = {"idx": 0}
    state["idx"] = next((i for i, (n, _) in enumerate(glyphs)
                         if n not in saved), len(glyphs))

    fig, ax = plt.subplots(figsize=(6, 7))
    plt.subplots_adjust(bottom=0.25)
    image_artist = ax.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=255)
    title = ax.set_title("")
    ax.axis("off")

    info_text = fig.text(0.5, 0.12,
                         "press any letter or digit to label, s to skip, "
                         "backspace to undo, q to save and quit",
                         ha="center", fontsize=9)

    def refresh():
        if state["idx"] >= len(glyphs):
            title.set_text("done, press q to save and quit")
            image_artist.set_data(np.zeros((28, 28)))
            fig.canvas.draw_idle()
            return
        name, img = glyphs[state["idx"]]
        image_artist.set_data(img)
        title.set_text(f"{state['idx'] + 1}/{len(glyphs)}  {name}  "
                       f"({len(saved)} labeled)")
        fig.canvas.draw_idle()

    def save_partial():
        with open(labels_path, "w") as f:
            json.dump(saved, f)

    def on_key(event):
        if event.key is None:
            return
        if event.key == "q":
            save_partial()
            print(f"saved partial labels to {labels_path}")
            plt.close()
            return
        if event.key == "backspace":
            if state["idx"] > 0:
                state["idx"] -= 1
                prev_name = glyphs[state["idx"]][0]
                if prev_name in saved:
                    del saved[prev_name]
            refresh()
            return
        if event.key == "s":
            state["idx"] += 1
            refresh()
            return
        ch = event.key
        if len(ch) == 1 and ch in VALID_CHARS:
            if state["idx"] < len(glyphs):
                name = glyphs[state["idx"]][0]
                saved[name] = ch
                state["idx"] += 1
                if state["idx"] % 25 == 0:
                    save_partial()
                refresh()

    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()

    # Final save and conversion to npz
    save_partial()
    write_npz(glyphs, saved, save_path, val_split)


def write_npz(glyphs, saved, save_path: str, val_split: float):
    classes = list(string.digits + string.ascii_lowercase + string.ascii_uppercase)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    images = []
    labels = []
    for name, img in glyphs:
        if name not in saved:
            continue
        if saved[name] not in class_to_idx:
            continue
        images.append(img)
        labels.append(class_to_idx[saved[name]])

    if not images:
        print("no labels saved")
        return

    images = np.stack(images).astype(np.uint8)
    labels = np.array(labels, dtype=np.int64)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(images))
    images = images[perm]
    labels = labels[perm]
    split = max(1, int((1 - val_split) * len(images)))

    out = Path(save_path)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "train.npz",
                        images=images[:split], labels=labels[:split])
    np.savez_compressed(out / "val.npz",
                        images=images[split:], labels=labels[split:])
    with open(out / "classes.txt", "w") as f:
        for c in classes:
            f.write(c + "\n")

    print(f"wrote {split} train + {len(images) - split} val samples to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glyphs", required=True,
                        help="folder of segmented glyph PNGs (from preprocess.py)")
    parser.add_argument("--out", default="data/custom",
                        help="output folder for labeled .npz dataset")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    try:
        label_session(args.glyphs, args.out, args.val_split)
    except KeyboardInterrupt:
        print("interrupted, partial labels preserved")
        sys.exit(0)
