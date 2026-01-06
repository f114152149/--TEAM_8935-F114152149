import cv2
from pathlib import Path

# ======================
# 你只需要改這兩個路徑
# ======================
IMAGES_ROOT = Path(r"outputs\train_aug\image")   # 你的圖片大資料夾
LABELS_ROOT = Path(r"outputs\train_aug\label")   # 你的標註大資料夾
OUT_ROOT    = Path("vis_out")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ======================
# YOLO 工具
# ======================
def read_yolo_label(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            cls = int(float(p[0]))
            x, y, w, h = map(float, p[1:])
            boxes.append((cls, x, y, w, h))
    return boxes

def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    xc, yc = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h
    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)
    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)
    return max(0,x1), max(0,y1), min(img_w-1,x2), min(img_h-1,y2)

def draw_boxes(img, boxes):
    h, w = img.shape[:2]
    for cls, x, y, bw, bh in boxes:
        x1,y1,x2,y2 = yolo_to_xyxy(x,y,bw,bh,w,h)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(
            img, str(cls), (x1, max(0,y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
        )
    return img

# ======================
# 主流程
# ======================
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in IMAGES_ROOT.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"找到圖片數量：{len(image_paths)}")

    drawn = 0
    missing = 0

    for img_path in image_paths:
        # label = 同資料夾、同檔名
        rel = img_path.relative_to(IMAGES_ROOT)
        label_path = (LABELS_ROOT / rel).with_suffix(".txt")

        if not label_path.exists():
            missing += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print("讀不到圖片：", img_path)
            continue

        boxes = read_yolo_label(label_path)
        vis = draw_boxes(img.copy(), boxes)

        out_path = (OUT_ROOT / rel).with_suffix(".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)

        drawn += 1

    print(f"✅ 完成：畫框 {drawn} 張")
    print(f"⚠️ 找不到 label 跳過 {missing} 張")

if __name__ == "__main__":
    main()
