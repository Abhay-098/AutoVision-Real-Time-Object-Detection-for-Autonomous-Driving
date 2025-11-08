import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms

class COCODataset(Dataset):
    def __init__(self, root, annotation_file):
        self.root = root
        self.annotation_file = annotation_file

        # Load COCO annotations
        with open(annotation_file) as f:
            data = json.load(f)

        self.images = {img["id"]: img["file_name"] for img in data["images"]}
        self.annotations = data["annotations"]

        # Group annotations by image ID
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Transform to convert PIL images to tensors
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = list(self.images.keys())[idx]
        img_path = os.path.join(self.root, self.images[image_id])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Load annotations for this image
        anns = self.img_to_anns.get(image_id, [])

        if len(anns) == 0:
            # Skip images with no bounding boxes
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = []
            labels = []
            for ann in anns:
                bbox = ann["bbox"]  # COCO format: [x, y, width, height]
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[0] + bbox[2]
                y_max = bbox[1] + bbox[3]
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann["category_id"])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        return img, target
