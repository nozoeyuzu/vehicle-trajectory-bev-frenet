#google colabでの実行を想定したコード

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch

# ====================================================
# 0. GPUが本当に使えているかのチェック
# ====================================================
if torch.cuda.is_available():
    print(f"✅ GPUが有効です: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️ GPUが使えません！CPUモードで動くため処理が非常に遅くなります。")
    print("上部メニューの『ランタイム』➔『ランタイムのタイプを変更』で『T4 GPU』を選択してください。")
    device = "cpu"

# ====================================================
# 1. 設定項目
# ====================================================
video_path = 'rakkabutu_highway.mp4'
csv_output_path = 'traffic_trajectory_data2.csv'

src_pts = np.float32([
    [635, 180],   # 0: 奥・左  → dst [0.0, 0.0]
    [710, 180],   # 1: 奥・右  → dst [3.5, 0.0]
    [565, 455],   # 2: 手前・左 → dst [0.0, 8.0]
    [735, 455],   # 3: 手前・右 → dst [3.5, 8.0]
])
dst_pts = np.float32([[0.0, 0.0], [3.5, 0.0], [0.0, 8.0], [3.5, 8.0]])

# ====================================================
# 2. 初期準備
# ====================================================
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
model = YOLO('yolo11m.pt') # 選択されたMediumモデル
cap = cv2.VideoCapture(video_path)
data_records = []

print("\n処理を開始します...")

# ====================================================
# 3. 動画のメインループ処理
# ====================================================
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 30フレーム（約1秒分）ごとに進捗を画面に出力する
    if frame_count % 30 == 0:
        print(f"現在 {frame_count} フレーム目を解析中...")

    # deviceを指定してGPUで確実に動かします
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=device)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, obj_id, cls_id in zip(boxes, ids, clss):
            if cls_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box
                foot_x = (x1 + x2) / 2.0
                foot_y = y2

                img_point = np.float32([[[foot_x, foot_y]]])
                bev_point = cv2.perspectiveTransform(img_point, matrix)

                data_records.append({
                    'Frame': frame_count,
                    'Vehicle_ID': obj_id,
                    'Class_ID': cls_id,
                    'X_pixel': foot_x,
                    'Y_pixel': foot_y,
                    'X_meter': bev_point[0][0][0],
                    'Y_meter': bev_point[0][0][1]
                })

cap.release()

# ====================================================
# 4. CSVファイルへの出力・保存
# ====================================================
if data_records:
    df = pd.DataFrame(data_records)
    df.to_csv(csv_output_path, index=False)
    print(f"\n🎉 完了！全 {frame_count} フレームの解析に成功しました。")
    print(f"保存先: {csv_output_path}")
else:
    print("\n❌ 車両が1台も検出されませんでした。動画ファイルや範囲設定（src_pts）を確認してください。")