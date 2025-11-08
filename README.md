# Faster R-CNN — AutoVision Example Project

This project provides a ready-to-run Faster R-CNN pipeline (using PyTorch / torchvision)
for object detection with scripts for training, evaluation, plotting metrics, and
utilities to use a custom dataset (VOC-style) or Pascal VOC / COCO datasets.

**What is included**
- `src/train.py` — training & validation loop (fine-tune Faster R-CNN).
- `src/evaluate.py` — evaluation script (attempts to use COCO mAP if pycocotools available,
   else computes simple IoU-based metrics).
- `src/dataset.py` — VOC-style dataset wrapper (for folders with images and Pascal VOC XML annotations).
- `src/utils.py` — helpers for visualization, plotting, checkpoints.
- `requirements.txt` — Python dependencies.
- `example_config.yaml` — example config for paths and hyperparams.
- `LICENSE` — MIT license.

**How to use (quick)**
1. Create a Python 3.8+ venv and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare data:
   * Either use Pascal VOC (download separately) or prepare a VOC-style dataset:
   ```text
   dataset/
     ├── JPEGImages/
     ├── Annotations/   # Pascal VOC XMLs
     └── ImageSets/Main/train.txt
   ```
3. Train:
   ```bash
   python src/train.py --data-root /path/to/dataset --num-classes 3 --output /path/to/exp
   ```
4. Evaluate & plot:
   ```bash
   python src/evaluate.py --data-root /path/to/dataset --weights /path/to/exp/ckpt_final.pth --output /path/to/exp
   ```

**Notes**
- The evaluation tries to use COCO-style mAP via `pycocotools`. If not installed, a fallback IoU-based AP is computed.
- For real training on GPUs, ensure `torch` is the CUDA build and a GPU is available.
- This project is a starting point — adapt dataset loading and augmentations for your use case.
