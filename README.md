# TEAM_8935 - YOLO 目標偵測專題

本專案包含資料增強、標註視覺化檢查、資料集整理與 YOLO 訓練、以及本機推論輸出等流程。

## 檔案說明
- `augmentation.py`：訓練影像資料增強（同步更新 YOLO 標註框），輸出到 `outputs/train_aug/`
- `test.py`：將 YOLO 標註畫回影像做抽樣檢查，輸出到 `vis_out/`
- `train.py`：病人層級資料切分、整理成 YOLO 資料結構並開始訓練
- `predict_local.py`：本機推論（分批 images1/images2）並輸出偵測結果文字檔
- `aortic_valve_colab.yaml`：Ultralytics YOLO 訓練資料設定檔

## 環境需求
- Python 3.x
- 套件：ultralytics、torch、opencv-python、albumentations

安裝（示例）：
```bash
pip install ultralytics torch opencv-python albumentations
資料夾放置（本 repo 不包含影像資料）
請自行準備並放置於專案根目錄：
training_image/：訓練影像（patient 子資料夾）
training_label/：訓練標註（YOLO .txt，結構對應 training_image）
testing_image/：測試影像（推論用）
執行順序
資料增強
python augmentation.py
增強資料畫框檢查
python test.py
資料整理與訓練
python train.py
本機推論（需有 best.pt）
python predict_local.py
重要提醒
train.py 預設讀取 training_image/ 與 training_label/，不會自動使用 outputs/train_aug/ 的增強資料。
predict_local.py 需要專案根目錄存在 best.pt 才能推論。
