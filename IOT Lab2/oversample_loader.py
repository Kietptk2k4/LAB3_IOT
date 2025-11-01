# oversample_loader.py
import json
from collections import defaultdict
from torch.utils.data import WeightedRandomSampler


def make_sampler(coco_json_path, incorrect_id=3, w_incorrect=6.0, w_normal=1.0):
    coco = json.load(open(coco_json_path, "r", encoding="utf-8"))
    ann_by_img = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a["category_id"])
    has_bad = {img["id"]: (incorrect_id in ann_by_img.get(img["id"], []))
               for img in coco["images"]}
    img_ids = [img["id"] for img in coco["images"]]
    weights = [w_incorrect if has_bad[i] else w_normal for i in img_ids]
    return WeightedRandomSampler(weights, num_samples=len(img_ids), replacement=True)
