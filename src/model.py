import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes: int):
    """
    Returns a Faster R-CNN model pre-trained on COCO,
    adjusted for the specified number of classes.
    """
    # Load pretrained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == "__main__":
    model = get_model_instance_segmentation(num_classes=3)
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Model loaded successfully on: {device}")
