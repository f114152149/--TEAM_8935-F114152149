import os
import shutil
import gc
from pathlib import Path

import torch
from ultralytics import YOLO

# ===============================
# 你要改的地方
# ===============================
TEST_ROOT = Path(r"testing_image")        # ← 改成你的 testing 影像「最外層資料夾」
BEST_PT   = Path(r"best.pt")      # ← 改成你的 best.pt 路徑
DEVICE    = 0                     # 0=GPU；沒GPU改成 "cpu"
IMGSZ     = 640

# ===============================
# 輸出資料夾
# ===============================
WORKDIR      = Path("datasets")
IMAGES1_DIR  = WORKDIR / "test" / "images1"
IMAGES2_DIR  = WORKDIR / "test" / "images2"
PRED_TXT_DIR = Path("predict_txt")

def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def collect_all_pngs_recursive(root: Path):
    # 遞迴抓所有 png/jpg/jpeg
    exts = (".png", ".jpg", ".jpeg")
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort(key=lambda x: x.name)  # 跟你 Colab 一樣用檔名排序
    return files

def split_copy_files(files, dst1: Path, dst2: Path):
    dst1.mkdir(parents=True, exist_ok=True)
    dst2.mkdir(parents=True, exist_ok=True)

    half = len(files) // 2
    for f in files[:half]:
        shutil.copy2(f, dst1 / f.name)
    for f in files[half:]:
        shutil.copy2(f, dst2 / f.name)
    return half

def write_results_txt(results, out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fout:
        for r in results:
            filename = Path(r.path).stem
            boxes = r.boxes
            if boxes is None:
                continue
            box_num = len(boxes.cls.tolist())
            if box_num == 0:
                continue
            for j in range(box_num):
                label = int(boxes.cls[j].item())
                conf  = float(boxes.conf[j].item())
                x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                fout.write(f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")

def merge_txt(file1: Path, file2: Path, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fout:
        for f in [file1, file2]:
            if f.exists():
                with open(f, "r", encoding="utf-8") as fin:
                    fout.writelines(fin.readlines())

def main():
    if not TEST_ROOT.exists():
        raise FileNotFoundError(f"找不到資料夾：{TEST_ROOT}")
    if not BEST_PT.exists():
        raise FileNotFoundError(f"找不到 best.pt：{BEST_PT}")

    ensure_clean_dir(IMAGES1_DIR)
    ensure_clean_dir(IMAGES2_DIR)
    ensure_clean_dir(PRED_TXT_DIR)

    all_files = collect_all_pngs_recursive(TEST_ROOT)
    if not all_files:
        raise RuntimeError(f"在 {TEST_ROOT} 找不到任何 png/jpg/jpeg")

    half = split_copy_files(all_files, IMAGES1_DIR, IMAGES2_DIR)
    print(f"來源資料夾：{TEST_ROOT}")
    print(f"總共 {len(all_files)} 張，前半 {half} 張 -> images1，後半 {len(all_files)-half} 張 -> images2")

    model = YOLO(str(BEST_PT))

    print("開始預測 images1 ...")
    results1 = model.predict(source=str(IMAGES1_DIR), save=True, imgsz=IMGSZ, device=DEVICE)
    write_results_txt(results1, PRED_TXT_DIR / "images1.txt")

    del results1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("開始預測 images2 ...")
    model = YOLO(str(BEST_PT))
    results2 = model.predict(source=str(IMAGES2_DIR), save=True, imgsz=IMGSZ, device=DEVICE)
    write_results_txt(results2, PRED_TXT_DIR / "images2.txt")

    merged = PRED_TXT_DIR / "merged.txt"
    merge_txt(PRED_TXT_DIR / "images1.txt", PRED_TXT_DIR / "images2.txt", merged)
    print(f"合併完成 -> {merged.resolve()}")
    print("✅ 全部完成")

if __name__ == "__main__":
    main()
