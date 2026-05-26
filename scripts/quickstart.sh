#!/usr/bin/env bash
# Full pipeline quickstart, from zero to a rendered "Hello World" sample.
#
# Steps:
#   1. install dependencies
#   2. generate the synthetic dataset (~20 seconds)
#   3. train the CNN (~4 minutes on CPU)
#   4. render a sample image of "Hello World" in the learned style
#
# Usage: bash scripts/quickstart.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/4] installing dependencies"
pip install -q -r requirements.txt

echo "[2/4] generating synthetic dataset"
python3 -m data.synthesize --samples 400

echo "[3/4] training the CNN (4 epochs, expect ~4 minutes on CPU)"
python3 -m models.train --epochs 4 --batch-size 256 --workers 0

echo "[4/4] rendering a sample"
python3 -m inference.generate --text "Hello World" --out output.png --seed 42

echo
echo "done. open output.png to see the rendered text."
