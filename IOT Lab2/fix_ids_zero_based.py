import json
import os

FILES = [
    r"D:\wordspace\IOT Lab2\dataset\train\_annotations.coco.json",
    r"D:\wordspace\IOT Lab2\dataset\valid\_annotations.coco.json",
]

# Mục tiêu: 0-based ids
TARGET = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2,
}


def fix_one(path):
    if not os.path.exists(path):
        print("Không thấy:", path)
        return
    d = json.load(open(path, "r", encoding="utf-8"))

    # 1) Chuẩn hoá categories về 0,1,2 + supercategory
    name2id_old = {c["name"]: c["id"] for c in d.get("categories", [])}
    idmap = {}
    for name, new_id in TARGET.items():
        old_id = name2id_old.get(name)
        if old_id is None:
            print(f"[WARN] Thiếu class '{name}' trong {path}")
            continue
        idmap[old_id] = new_id

    d["categories"] = [
        {"id": TARGET["with_mask"], "name": "with_mask",
            "supercategory": "none"},
        {"id": TARGET["without_mask"],
            "name": "without_mask", "supercategory": "none"},
        {"id": TARGET["mask_weared_incorrect"],
            "name": "mask_weared_incorrect", "supercategory": "none"},
    ]

    # 2) Sửa toàn bộ annotations.category_id theo idmap
    changed = 0
    skipped = 0
    for ann in d.get("annotations", []):
        cid = ann.get("category_id")
        if cid in idmap:
            new_cid = idmap[cid]
            if new_cid != cid:
                ann["category_id"] = new_cid
                changed += 1
        else:
            # Nếu gặp id lạ, cố map theo tên nếu có, còn không thì bỏ qua
            skipped += 1

    # 3) (khuyến nghị) chuẩn hoá đường dẫn ảnh về "images/<tên>"
    for im in d.get("images", []):
        base = os.path.basename(im.get("file_name", ""))
        im["file_name"] = f"images/{base}"

    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Đã vá: {path} | sửa {changed} annotation(s), bỏ qua {skipped}")


for p in FILES:
    fix_one(p)
