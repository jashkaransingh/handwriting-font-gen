# handwriting-font-gen

trains a CNN on your own handwriting and renders any text in your hand. custom data pipeline cut training prep from 3 hours to 15 minutes.

## pipeline

```
raw handwriting photos
   │
   ▼  preprocess.py
segmented character images (normalized, denoised)
   │
   ▼  label_gui.py
labeled dataset (character to image mapping)
   │
   ▼  augment.py
augmented training set (rotations, noise, morphology)
   │
   ▼  train.py
trained CNN checkpoint
   │
   ▼  generate.py
text rendered in your handwriting
```

## run it

```bash
git clone https://github.com/jashkaransingh/handwriting-font-gen
cd handwriting-font-gen
pip install -r requirements.txt

# step 1, preprocess scanned handwriting sheets
python data/preprocess.py --input raw/ --output processed/

# step 2, label characters with the GUI
python data/label_gui.py --input processed/ --output labeled/

# step 3, train
python models/train.py --data labeled/ --epochs 50 --output checkpoints/

# step 4, generate
python inference/generate.py --model checkpoints/best.pth --text "Hello World"
```

## the hard part

training prep used to take 3 hours per run because OpenCV preprocessing was running serially over thousands of images, one at a time. parallelized it with multiprocessing, batched the augmentation step (rotations, morphological ops, synthetic noise), and built a small Matplotlib labeling GUI because relabeling by hand was destroying me. dropped prep time to 15 minutes. the actual CNN was the easy part once data flowed cleanly.

## results

- training prep dropped from 3 hours to 15 minutes
- character recognition accuracy at 94.2% on held-out validation
- full sentences render with proper spacing and baseline alignment

## stack

Python, PyTorch, OpenCV, NumPy, scikit-learn for the baseline, Matplotlib for the GUI.
