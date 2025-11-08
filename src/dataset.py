import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET

class VOCDataset(torch.utils.data.Dataset):
    """A minimal Pascal VOC-style dataset wrapper.
    Expects folders:
      - JPEGImages/     (images)
      - Annotations/    (VOC XMLs)
      - ImageSets/Main/train.txt (list of image ids)
    """
    def __init__(self, root, image_set='train', transforms=None, classes=None):
        self.root = root
        self.image_set = image_set
        self.transforms = transforms
        ids_file = os.path.join(root, 'ImageSets', 'Main', image_set + '.txt')
        with open(ids_file) as f:
            self.ids = [x.strip() for x in f.readlines()]
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.ann_dir = os.path.join(root, 'Annotations')
        # classes: list like ['__background__','car','person']
        if classes is None:
            self.classes = ['__background__','object']
        else:
            self.classes = classes

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, img_id + '.jpg')
        ann_path = os.path.join(self.ann_dir, img_id + '.xml')
        img = Image.open(img_path).convert('RGB')
        boxes = []
        labels = []
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.classes:
                # map unknown classes to background (skip)
                continue
            labels.append(self.classes.index(name))
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
