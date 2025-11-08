#!/bin/bash
echo "============================================"
echo "🚀 AutoVision Faster R-CNN Training Started"
echo "============================================"

DATA_ROOT="./dataset/export"
OUTPUT="./output"
NUM_CLASSES=3

if [ ! -d "$OUTPUT" ]; then
  mkdir -p "$OUTPUT"
fi

if command -v nvidia-smi &> /dev/null; then
  DEVICE="GPU"
else
  DEVICE="CPU"
fi

echo "🖥️ Detected Device: $DEVICE"
python src/train.py --data-root $DATA_ROOT --num-classes $NUM_CLASSES --output $OUTPUT

# Auto-open loss graph if exists
if [ -f "$OUTPUT/loss_plot.png" ]; then
  echo "📊 Opening training loss graph..."
  xdg-open "$OUTPUT/loss_plot.png" >/dev/null 2>&1 || open "$OUTPUT/loss_plot.png"
else
  echo "⚠️ Loss graph not found."
fi
