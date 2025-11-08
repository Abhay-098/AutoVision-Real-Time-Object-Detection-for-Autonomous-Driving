import argparse
import os
import torch
import torchvision
from dataset import VOCDataset
from utils import load_checkpoint, visualize_predictions, plot_metrics
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

def get_model(num_classes, weights_path=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if weights_path:
        chk = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(chk['model'])
    return model

def compute_iou(boxA, boxB):
    # box: [xmin,ymin,xmax,ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def simple_eval(model, dataset, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    all_scores = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, target = dataset[idx]
            img_t = torchvision.transforms.functional.to_tensor(img).to(device)
            pred = model([img_t])[0]
            gt_boxes = target['boxes'].numpy() if 'boxes' in target else []
            gt_labels = target['labels'].numpy() if 'labels' in target else []
            matched_gt = set()
            for box_pred, label_pred, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                if score < score_threshold:
                    continue
                box_pred = box_pred.cpu().numpy()
                label_pred = label_pred.cpu().numpy()
                best_iou = 0; best_j = -1
                for j, gt in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = compute_iou(box_pred, gt)
                    if iou > best_iou:
                        best_iou = iou; best_j = j
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_j)
                else:
                    fp += 1
                all_scores.append(score.cpu().numpy() if hasattr(score,'cpu') else float(score))
            fn += max(0, len(gt_boxes) - len(matched_gt))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--output', default='./output')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.num_classes, args.weights)
    model.to(device)
    dataset = VOCDataset(args.data_root, image_set='val', classes=None)
    metrics = simple_eval(model, dataset, device)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output,'metrics.txt'),'w') as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v}\n")
    print('Evaluation results:')
    print(metrics)
