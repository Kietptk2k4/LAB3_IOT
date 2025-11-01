# fix_filenames_to_images_only.py
import json
import os

FILES = [
    r"D:\wordspace\IOT Lab2\dataset\train\_annotations.coco.json",
    r"D:\wordspace\IOT Lab2\dataset\valid\_annotations.coco.json",
]


def fix_one(path):
    if not os.path.exists(path):
        print("Không thấy:", path)
        return
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    changed = 0
    for im in d.get("images", []):
        fn = im.get("file_name", "")
        base = os.path.basename(fn)
        # luôn chuẩn hóa về "images/<basename>"
        new_fn = f"images/{base}"
        if fn != new_fn:
            im["file_name"] = new_fn
            changed += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print(f"Đã vá: {path} | đổi {changed} file_name")


for p in FILES:
    fix_one(p)
