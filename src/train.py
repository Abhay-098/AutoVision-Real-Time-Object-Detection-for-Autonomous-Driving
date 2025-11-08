import argparse
import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import VOCDataset
from utils import save_checkpoint, plot_metrics
from tqdm import tqdm

def get_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = VOCDataset(args.data_root, image_set='train', classes=args.classes)
    dataset_val = VOCDataset(args.data_root, image_set='val', classes=args.classes)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(args.num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {'loss':[], 'val_loss':[], 'map':[]}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        it = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
        for images, targets in pbar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value
            it += 1
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            history['loss'].append(loss_value)
            pbar.set_postfix(loss=loss_value)
        lr_scheduler.step()
        # save checkpoint
        save_checkpoint({'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch},
                        os.path.join(args.output, f'ckpt_epoch_{epoch}.pth'))
        # simple validation loop to compute average validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in data_loader_val:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        val_loss = val_loss / max(1, len(data_loader_val))
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch} finished. Train loss={epoch_loss/it:.4f}, Val loss={val_loss:.4f}")
    # final save
    save_checkpoint({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': args.epochs},
                    os.path.join(args.output, 'ckpt_final.pth'))
    plot_metrics(history, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='path to VOC-style dataset root')
    parser.add_argument('--num-classes', type=int, required=True, help='number of classes (including background)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--classes', nargs='+', default=['__background__','object'])
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    train(args)
