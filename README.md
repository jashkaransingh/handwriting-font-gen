# handwriting-font-gen

Trains a CNN on images of your own handwriting and generates text that looks like you wrote it. Custom data pipeline cut training prep time from ~3 hours to 15 minutes.

## Pipeline

```
Raw handwriting photos
        │
        ▼  preprocess.py
Segmented character images (normalized, denoised)
        │
        ▼  label_gui.py
Labeled dataset (character → image mapping)
        │
        ▼  augment.py
Augmented training set (rotations, noise, morphology)
        │
        ▼  train.py
Trained CNN model checkpoint
        │
        ▼  generate.py
"Hello World" → rendered in your handwriting
```

## Quick start

```bash
git clone https://github.com/jashkaransingh/handwriting-font-gen
cd handwriting-font-gen
pip install -r requirements.txt

# Step 1: preprocess your scanned handwriting sheets
python data/preprocess.py --input raw/ --output processed/

# Step 2: label characters with the GUI
python data/label_gui.py --input processed/ --output labeled/

# Step 3: train
python models/train.py --data labeled/ --epochs 50 --output checkpoints/

# Step 4: generate text
python inference/generate.py --model checkpoints/best.pth --text "Hello World"
```

## Results

- Training prep: **3 hours → 15 minutes** after pipeline optimization
- Character recognition accuracy: **94.2%** on held-out validation set
- Renders full sentences with proper spacing and baseline alignment
