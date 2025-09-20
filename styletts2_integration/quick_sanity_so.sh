#!/usr/bin/env bash
# Run a quick sanity check by synthesising a handful of Somali sentences.
# Ensure that `styletts2_integration/tts_infer_so.py` is executable and that
# you have trained StyleTTS2 with the provided configs.  Modify the
# --styletts2_checkpoint path to point to your trained checkpoint.

set -e

PLBERT_DIR="runs/plbert_so/packaged"
CHECKPOINT="path/to/your/styletts2_stage2_checkpoint.pt"
OUT_DIR="out/sanity_test"

python styletts2_integration/tts_infer_so.py \
  --text "Salaan! Sidee tahay?" \
  --plbert_dir "$PLBERT_DIR" \
  --styletts2_checkpoint "$CHECKPOINT" \
  --out "$OUT_DIR"

python styletts2_integration/tts_infer_so.py \
  --text "Waxaan jeclahay buugaagta." \
  --plbert_dir "$PLBERT_DIR" \
  --styletts2_checkpoint "$CHECKPOINT" \
  --out "$OUT_DIR"

python styletts2_integration/tts_infer_so.py \
  --text "Tani waa imtixaan degdeg ah." \
  --plbert_dir "$PLBERT_DIR" \
  --styletts2_checkpoint "$CHECKPOINT" \
  --out "$OUT_DIR"