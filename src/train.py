import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import COCODataset
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from PIL import Image

# -----------------------------
# Build Model
# -----------------------------
def get_model_instance_segmentation(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# -----------------------------
# Clip bounding boxes to image size
# -----------------------------
def clip_boxes(boxes, width, height):
    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=width)
    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=height)
    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=width)
    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=height)
    return boxes

# -----------------------------
# Training Function
# -----------------------------
def train(args):
    print("Using device:", args.device)
    print("Loading COCO dataset...")

    annotation_file = os.path.join(args.data_root, "annotations.coco.json")
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"❌ Annotation file not found: {annotation_file}")

    dataset = COCODataset(args.data_root, annotation_file)

    # Use small subset for debugging
    if args.debug:
        subset_size = min(100, len(dataset))
        dataset = Subset(dataset, list(range(subset_size)))
        print(f"⚠️ Using a debug subset of {subset_size} images for fast training")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model_instance_segmentation(args.num_classes)
    model.to(args.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    print(f"📦 Dataset size: {len(dataset)} images")
    print(f"🧠 Training on {args.device} for {args.epochs} epochs...")

    train_losses = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")

        for images, targets in progress_bar:
            # Skip images with zero boxes
            non_empty_targets = []
            non_empty_images = []
            for img, t in zip(images, targets):
                if t["boxes"].shape[0] > 0:
                    # Clip boxes to image size
                    width, height = img.shape[2], img.shape[1]
                    t["boxes"] = clip_boxes(t["boxes"], width, height)

                    # Warn if category IDs exceed num_classes
                    if t["labels"].max().item() >= args.num_classes:
                        print(f"⚠️ Warning: label {t['labels'].max().item()} >= num_classes {args.num_classes}")

                    non_empty_targets.append({k: v.to(args.device) for k, v in t.items()})
                    non_empty_images.append(img.to(args.device))

            if len(non_empty_images) == 0:
                continue  # skip batch with no valid boxes

            loss_dict = model(non_empty_images, non_empty_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()
            epoch_loss += loss_value
            progress_bar.set_postfix(loss=loss_value)

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"📉 Epoch [{epoch+1}/{args.epochs}] - Average Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "fasterrcnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model saved to {model_path}")

    # Plot loss
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', color='b')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    loss_plot_path = os.path.join(args.output, "loss_plot.png")
    plt.savefig(loss_plot_path)
    print(f"📊 Loss curve saved to {loss_plot_path}")
    plt.close()

    print("\n--------------------------------------------")
    print("✅ Training complete!")
    print(f"📁 Results saved to: {args.output}")
    print("--------------------------------------------")

# -----------------------------
# Main Function
# -----------------------------
if __name__ == "__main__":
    print("============================================")
    print("🚀 AutoVision Faster R-CNN Training Started")
    print("============================================")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to dataset root")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes (including background)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0005)  # smaller LR to prevent exploding loss
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--debug", action="store_true", help="Use small subset of dataset for debugging")
    args = parser.parse_args()

    # Auto-detect GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🖥️ Using device: {args.device}")
    print(f"📦 Dataset: {args.data_root}")
    print(f"🕐 Training for {args.epochs} epochs")
    print("--------------------------------------------")

    train(args)
