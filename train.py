import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ===============================
# 1) 你的本機資料路徑（改這裡）
# ===============================
SRC_IMG_ROOT = Path("training_image")   # 例如 Path(r"C:\...\training_image")
SRC_LBL_ROOT = Path("training_label")   # 例如 Path(r"C:\...\training_label")
YAML_PATH    = Path("aortic_valve_colab.yaml")  # 例如 Path(r"C:\...\aortic_valve_colab.yaml")

# patient 切分
TRAIN_PATIENTS = range(1, 31)   # patient0001~0030
VAL_PATIENTS   = range(31, 51)  # patient0031~0050

# 輸出 YOLO dataset 位置
DST_ROOT = Path("datasets")
TRAIN_IMG_DIR = DST_ROOT / "train" / "images"
TRAIN_LBL_DIR = DST_ROOT / "train" / "labels"
VAL_IMG_DIR   = DST_ROOT / "val" / "images"
VAL_LBL_DIR   = DST_ROOT / "val" / "labels"

# ===============================
# 2) 工具：建立乾淨資料夾
# ===============================
def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

# ===============================
# 3) 依 patient 取得排序後的影像與標註清單
#    (解決你圖片 0001、label 0201 這種檔名不一致問題)
# ===============================
def list_images(patient_dir: Path):
    exts = (".png", ".jpg", ".jpeg")
    files = [f for f in patient_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
    return sorted(files, key=lambda x: x.name)

def list_labels(patient_dir: Path):
    files = [f for f in patient_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    return sorted(files, key=lambda x: x.name)

def copy_patient(patient_idx: int, split: str):
    patient = f"patient{patient_idx:04d}"
    img_dir = SRC_IMG_ROOT / patient
    lbl_dir = SRC_LBL_ROOT / patient

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        print(f"⚠️ 找不到資料夾，跳過：{patient}")
        return 0

    imgs = list_images(img_dir)
    lbls = list_labels(lbl_dir)

    if not imgs or not lbls:
        print(f"⚠️ {patient} 圖或標註為空 (img={len(imgs)}, lbl={len(lbls)})，跳過")
        return 0

    n = min(len(imgs), len(lbls))
    if len(imgs) != len(lbls):
        print(f"⚠️ {patient} 數量不一致：img={len(imgs)} lbl={len(lbls)} → 只配對前 {n} 筆")

    if split == "train":
        out_img_dir, out_lbl_dir = TRAIN_IMG_DIR, TRAIN_LBL_DIR
    else:
        out_img_dir, out_lbl_dir = VAL_IMG_DIR, VAL_LBL_DIR

    count = 0
    for i in range(n):
        img_path = imgs[i]
        lbl_path = lbls[i]

        # 讓輸出檔名用「圖片檔名」當主（YOLO 最常見）
        out_img = out_img_dir / img_path.name
        out_lbl = out_lbl_dir / (img_path.stem + ".txt")

        shutil.copy2(img_path, out_img)
        shutil.copy2(lbl_path, out_lbl)
        count += 1

    return count

# ===============================
# 4) 主流程：建立 datasets 並搬資料
# ===============================
def main():
    # 基本檢查
    if not SRC_IMG_ROOT.exists():
        raise FileNotFoundError(f"找不到影像資料夾：{SRC_IMG_ROOT}")
    if not SRC_LBL_ROOT.exists():
        raise FileNotFoundError(f"找不到標註資料夾：{SRC_LBL_ROOT}")
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"找不到 YAML：{YAML_PATH}")

    ensure_clean_dir(TRAIN_IMG_DIR)
    ensure_clean_dir(TRAIN_LBL_DIR)
    ensure_clean_dir(VAL_IMG_DIR)
    ensure_clean_dir(VAL_LBL_DIR)

    train_cnt = 0
    val_cnt = 0

    for i in TRAIN_PATIENTS:
        train_cnt += copy_patient(i, "train")

    for i in VAL_PATIENTS:
        val_cnt += copy_patient(i, "val")

    print("✅ 資料整理完成")
    print("Train pairs:", train_cnt)
    print("Val pairs:", val_cnt)

    # ===============================
    # 5) YOLO 訓練
    # ===============================
    model = YOLO("yolo12n.pt")
    model.train(
        data=str(YAML_PATH),
        epochs=20,
        batch=16,
        imgsz=640,
        device=0
    )

    print("✅ 訓練完成（第一階段）")

if __name__ == "__main__":
    main()
