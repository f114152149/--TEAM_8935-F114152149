import os
import cv2
import glob
import random
from pathlib import Path

import albumentations as A

# =========================
# 路徑設定
# =========================
IN_IMG_DIR = "training_image"
IN_LBL_DIR = "training_label"

OUT_IMG_DIR = "outputs/train_aug/image"
OUT_LBL_DIR = "outputs/train_aug/label"

AUGS_PER_IMAGE = 2   # 每張原圖產生幾張增強圖（可改）
SEED = 42

random.seed(SEED)

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# =========================
# 讀 YOLO label
# =========================
def read_yolo_label(label_path: str):
    class_ids = []
    bboxes = []
    if not os.path.exists(label_path):
        return class_ids, bboxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            class_ids.append(cls)
            bboxes.append([x, y, w, h])
    return class_ids, bboxes


def write_yolo_label(label_path: str, class_ids, bboxes):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, (x, y, w, h) in zip(class_ids, bboxes):
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# =========================
# 定義增強 pipeline
# =========================
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_ids"],
        min_visibility=0.20
    )
)

# =========================
# 主程式：遞迴抓所有圖片
# =========================
img_paths = sorted(
    glob.glob(os.path.join(IN_IMG_DIR, "**", "*.png"), recursive=True) +
    glob.glob(os.path.join(IN_IMG_DIR, "**", "*.jpg"), recursive=True) +
    glob.glob(os.path.join(IN_IMG_DIR, "**", "*.jpeg"), recursive=True)
)

if not img_paths:
    raise FileNotFoundError(f"找不到圖片：{IN_IMG_DIR}")

print(f"找到 {len(img_paths)} 張訓練圖片（包含子資料夾）")

missing_labels = 0

for img_path in img_paths:
    # 取得相對路徑（例如：patient0051/xxx.png）
    rel_path = os.path.relpath(img_path, IN_IMG_DIR)
    rel_dir = os.path.dirname(rel_path)  # patient0051
    img_name = Path(img_path).stem

    # label 用相同相對路徑去找（patient0051/xxx.txt）
    lbl_path = os.path.join(IN_LBL_DIR, rel_dir, img_name + ".txt")
    if not os.path.exists(lbl_path):
        missing_labels += 1

    image = cv2.imread(img_path)
    if image is None:
        print("讀不到圖片，跳過：", img_path)
        continue

    class_ids, bboxes = read_yolo_label(lbl_path)

    for k in range(AUGS_PER_IMAGE):
        augmented = transform(image=image, bboxes=bboxes, class_ids=class_ids)

        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_class_ids = augmented["class_ids"]

        out_img_name = f"{img_name}_aug{k}.png"

        # 輸出也保留相同子資料夾結構
        out_img_path = os.path.join(OUT_IMG_DIR, rel_dir, out_img_name)
        out_lbl_path = os.path.join(OUT_LBL_DIR, rel_dir, f"{img_name}_aug{k}.txt")

        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        cv2.imwrite(out_img_path, aug_img)
        write_yolo_label(out_lbl_path, aug_class_ids, aug_bboxes)

print("✅ 資料增強完成")
print("輸出到：", OUT_IMG_DIR, OUT_LBL_DIR)
print(f"⚠️ 找不到對應 label 的圖片數量：{missing_labels}")
