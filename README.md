# AutoVision – Real-Time Object Detection for Autonomous Driving

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)](https://pytorch.org/)

---

## 🚀 Project Overview

**AutoVision** is a real-time object detection system for autonomous driving using **Faster R-CNN**.

It detects multiple road objects (vehicles, pedestrians, traffic signs, etc.) from camera images.

Key features:

* Trains on **COCO-format datasets**
* Customizable **number of classes**
* **Debug mode** for fast testing on a subset
* Automatic **loss visualization** and checkpointing

---

## 📁 Dataset Structure

Your dataset should look like this:

```
dataset/
└── export/
    ├── 00001.jpg
    ├── 00002.jpg
    └── annotations.coco.json
```

* `annotations.coco.json` must follow **COCO format**.
* Images can be directly in the `export` folder.

> ⚠️ Make sure all `category_id` values ≤ `--num-classes - 1`.

---

## ⚙️ Installation

1. Clone repository:

```bash
git clone https://github.com/Abhay-098/AutoVision-Object-Detection.git
cd AutoVision-Object-Detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

### Full dataset

```bash
python src/train.py --data-root ./dataset/export --num-classes 12 --output ./output
```

### Debug mode (fast test, first 100 images)

```bash
python src/train.py --data-root ./dataset/export --num-classes 12 --output ./output --debug
```

**Notes:**

* `--epochs` : Number of epochs (default 10)
* `--batch-size` : Batch size (default 2)
* `--lr` : Learning rate (default 0.0005 recommended)

---

## 💾 Output

After training, `./output` contains:

```
output/
├── fasterrcnn_model.pth       # Trained model weights
├── loss_plot.png              # Training loss curve
```

**Example: Loss Curve**

![Loss Curve](./screenshots/loss_plot.png)

---

## 🖼️ Inference / Detection

Load your trained model and run on new images:

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
num_classes = 12
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("./output/fasterrcnn_model.pth", map_location="cpu"))
model.eval()

# Load image
image = Image.open("dataset/export/00001.jpg").convert("RGB")
transform = T.ToTensor()
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)

# Draw boxes (example placeholder)
print(outputs)
```

**Example: Detection Output**

![Detection Example](./screenshots/detection_example.png)

> Replace these placeholders with screenshots from your training and inference.

---

## ⚡ Tips

* **GPU recommended** for full dataset. CPU training is very slow.
* Always run `--debug` first to check for exploding losses or annotation errors.
* Ensure all bounding boxes are within image bounds.

---

## 📄 References

* [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
* [PyTorch Detection Models](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
* [COCO Dataset Format](https://cocodataset.org/#format-data)

---

## 👨‍💻 Author

**Abhay Kumar** – [GitHub](https://github.com/Abhay-098)

---

### ✅ Next Steps

1. Run **debug training** → check loss curve.
2. Run **full training** → save model weights.
3. Add **screenshots**:

```
screenshots/
├── loss_plot.png
└── detection_example.png
```

> Use `matplotlib` or your inference script to capture these images.
