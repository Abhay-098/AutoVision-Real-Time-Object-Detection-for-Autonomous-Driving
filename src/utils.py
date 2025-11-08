import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision
from PIL import Image, ImageDraw

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None):
    chk = torch.load(path, map_location='cpu')
    model.load_state_dict(chk['model'])
    if optimizer and 'optimizer' in chk:
        optimizer.load_state_dict(chk['optimizer'])
    return chk

def visualize_predictions(model, device, image_path, classes, out_path=None, score_threshold=0.5):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_t = F.to_tensor(img).to(device)
    with torch.no_grad():
        preds = model([img_t])[0]
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
        if score < score_threshold:
            continue
        xmin, ymin, xmax, ymax = box.tolist()
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
        draw.text((xmin, ymin), f"{classes[label]}: {score:.2f}")
    if out_path:
        img.save(out_path)
    return img

def plot_metrics(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # history: dict with lists: loss, val_loss, map, val_map
    if 'loss' in history:
        plt.figure()
        plt.plot(history['loss'])
        plt.title('Train Loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig(os.path.join(out_dir,'train_loss.png'))
        plt.close()
    if 'val_loss' in history:
        plt.figure()
        plt.plot(history['val_loss'])
        plt.title('Val Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(out_dir,'val_loss.png'))
        plt.close()
    if 'map' in history:
        plt.figure()
        plt.plot(history['map'])
        plt.title('mAP')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.savefig(os.path.join(out_dir,'map.png'))
        plt.close()
