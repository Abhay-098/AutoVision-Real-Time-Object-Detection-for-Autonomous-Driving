import json

ann_file = "./dataset/export/annotations.coco.json"

with open(ann_file) as f:
    data = json.load(f)

max_cat = max(ann["category_id"] for ann in data["annotations"])
print("Max category_id:", max_cat)
print("Recommended num_classes:", max_cat + 1)
