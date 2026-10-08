# -*- coding: utf-8 -*-
"""
assign_lane_occupancy.py — 車両ごとの代表値・クラス（大きさ基準）・車線を決める

使い方:
    python compare_bbox2/assign_lane_occupancy.py P33C01      （frenet_sweep.py の後）

なぜ必要か:
    斜めの画角では、大型車の基準点が f によって約1車線分も横に動く。
    f=0.5 などの「評価したい f」で車線を決めると、大型車が隣の車線に割り当てられ、
    クラス間差が折り返されて小さく見え、f* もその f に引き寄せられる（循環）。
    そこで大型車の車線は f に依存しない「占有」で決める。

手順:
    1. 車線中心   : 普通車の d 中央値（f=0.5）のヒストグラムの山。中央分離帯（間隔 > 車線間隔の中央値 × CARRIAGEWAY_GAP_RATIO）で
                    車道を分け、中心線（d=0）を含む側＝ホモグラフィの車道だけを評価する
    2. クラス     : 同じ車線の普通車の bbox 中央値に対する大きさ size = √(幅比×高さ比)
                    car    = Class 2 かつ size < SIZE_CAR_MAX
                    pickup = Class 5/7 かつ size < SIZE_CAR_MAX（COCO の truck にはピックアップが含まれる）
                    large  = Class 5/7 かつ size ≥ SIZE_LARGE_MIN
    3. 普通車の車線: 小さい bbox は f による横ずれが小さいので f=0.5 の最寄り車線
    4. 大型車の車線: f=0, 0.5, 1 での最寄り車線が全部同じなら、その車線（真横の画角はほぼこれ）。
                    違えば占有で判定する:
                      各フレームで、候補車線ごとに「トラック車体と確実に重なる位置にいる基準車」を数え、
                      1車線だけ衝突がほぼ無く（≤ MAX_SELF フレーム）、他の候補は明確に衝突している
                      （≥ K_CONFLICT フレーム かつ 別々の基準車 ≥ MIN_CONFLICT_CARS 台）ならその車線。
                      どちらでもなければ判定不能（評価から除外）
                      ※ 台数の条件は渋滞対策（停止中は隣の1台だけで衝突フレームが数十に膨らむ）。初期値は1台
                    車体の範囲: bbox 下辺が車体のどちらの端に当たるかは画角で決まる
                      近づく画角 = 前端、遠ざかる画角 = 後端、真横 = 車体全長
                    端の位置は f=0〜1 のどこかなので、その不確かさを除いた「確実に車体がある区間」だけで数える
                    既知の限界: YOLO がトレーラーだけを truck、前のトラクターを別の「普通車」として検出すると、
                    トラクターが「本当の車線にいる普通車」として数えられ、隣の車線と誤判定されることがある
                    （遠ざかる画角のセミトレーラーで確認。対策を2通り試したが効果が不十分だったため入れていない）

出力（out_<CSV名>2/）:
    <CSV名>_vehicles.csv  車両ごとの代表値（f ごとの d 平均・標準偏差）、group、lane、lane_method、衝突数
    <CSV名>_lanes.csv     車線ごとの中心・カメラからの横距離・カメラ側の符号
"""

import sys
import numpy as np
import pandas as pd

import frenet_transform as ft
from common import CAMERAS, DEFAULT_CAMERAS, F_VALUES, out_dir, out_path, col, load_H, camera_position
import os

# ============== 設定 ==============
MIN_FRAMES        = 20     # これ未満の車両は除外
LANE_CHANGE_M     = 2.0    # 追跡前半1/4と後半1/4の d 中央値の差が「どの f でも」これを超えたら車線変更として除外
                           # （f=0.5 だけで判定すると、遠ざかる大型車は bbox の形の変化で点が横に流れ、
                           #   車線変更していないのに除外される。本当の車線変更はどの f でも d が変わる）
CARRIAGEWAY_GAP_RATIO = 1.5   # 車線中心の間隔が「間隔の中央値」のこの倍数を超えたら中央分離帯とみなす
                              # （固定の距離にすると、分離帯の狭い道路で車道を分けられない）
REF_OFF_M         = 1.0    # 占有判定の基準車: 車線中心からこの範囲内の普通車サイズの車両
LANE_HALF_M       = 1.85   # 普通車・ピックアップの車線割り当ての許容（半車線）
SIZE_CAR_MAX      = 1.4    # これ未満 = 普通車サイズ
SIZE_LARGE_MIN    = 1.6    # これ以上 = 大型車（1.4〜1.6 は判定せず除外）
TRUCK_LEN_M       = 16.0   # 占有判定で仮定する大型車の長さ [m]。長すぎると短いトラックの前後の車を衝突と数える
                           # （10〜20 m で f* はほぼ同じ。短いほど判定できる台数が減る）
MARGIN_M          = 2.0    # 車体区間の両端から除く余裕 [m]（基準車の基準点のずれ分）
MAX_INSIDE        = 0.5    # 基準車の bbox がこの割合以上トラックの bbox に含まれるフレームは数えない
                           # （トラックの一部を普通車として二重検出したものを衝突と数えないため）
K_CONFLICT        = 5      # 「明確に衝突」とみなすフレーム数
MAX_SELF          = 1      # 正解車線で許す衝突フレーム数（ノイズ分）
MIN_CONFLICT_CARS = 1      # 「明確に衝突」に必要な別々の基準車の台数。基準車は安定した追跡に絞っているので1台で十分
                           # （2にすると5分の動画では判定できる大型車がほぼ無くなった。長い動画で渋滞が多いときに検討）
CONGEST_KMH       = 60.0   # 車両の平均速度がこれ未満なら渋滞中（congested）、以上なら自由走行（free）
MOTION            = None   # "approach" / "recede" / "side"。None で軌跡から自動判定
SIDE_TAN          = np.tan(np.radians(20))   # 画像上の進行方向がこれより水平に近ければ真横
# ==================================


def _shift(d):
    q = max(3, len(d) // 4)
    return abs(np.median(d[-q:]) - np.median(d[:q]))


def per_vehicle(df):
    df = df.sort_values(["Vehicle_ID", "Frame"])
    df = df.assign(w=df.BBox_x2 - df.BBox_x1, h=df.BBox_y2 - df.BBox_y1)
    g = df.groupby("Vehicle_ID")
    v = pd.DataFrame({
        "cls":   g.Class_ID.agg(lambda s: s.mode().iloc[0]),
        "n":     g.size(),
        "w":     g.w.median(),
        "h":     g.h.median(),
        "s_med": g[col("s", 0.5)].median(),
        "d_med": g[col("d", 0.5)].median(),
        "dx_px": g.X_pixel.agg(lambda s: s.iloc[-1] - s.iloc[0]),
        "dy_px": g.Y_pixel.agg(lambda s: s.iloc[-1] - s.iloc[0]),
    })
    # 速度 [km/h]: s(f=0.5) を時刻に直線当てはめした傾き（最小二乗の傾きを集計値から計算）
    t, sv = df.Time_sec, df[col("s", 0.5)]
    a = pd.DataFrame({"t": t, "s": sv, "ts": t * sv, "tt": t * t, "id": df.Vehicle_ID}).groupby("id").mean()
    v["speed_kmh"] = (np.abs((a.ts - a.t * a.s) / (a.tt - a.t ** 2)) * 3.6).reindex(v.index)
    # 車線変更量: f ごとの「追跡前半→後半の d の変化」のうち最小のもの（f に依存しない判定）
    shifts = np.column_stack([g[col("d", f)].agg(lambda s: _shift(s.to_numpy())).reindex(v.index)
                              for f in F_VALUES])
    v["d_shift"] = shifts.min(axis=1)
    v["d_shift_f05"] = shifts[:, F_VALUES.index(0.5)]   # 参考: 以前の判定（f=0.5 のみ）
    for f in F_VALUES:
        v[col("dm", f)] = g[col("d", f)].mean()     # 代表値（車両内平均）
        v[col("dsd", f)] = g[col("d", f)].std()     # 車両内のばらつき（軌跡の綺麗さ）
    return v


def nearest(values, centers):
    i = np.abs(np.asarray(values)[:, None] - centers[None, :]).argmin(1)
    return i, np.asarray(values) - centers[i]


def detect_motion(v):
    car = v[(v.cls == 2) & (np.hypot(v.dx_px, v.dy_px) > 50)]
    ratio = np.median(np.abs(car.dy_px) / (np.abs(car.dx_px) + 1e-9))
    if ratio < SIDE_TAN:
        return "side"
    return "approach" if np.median(car.dy_px) > 0 else "recede"   # 画像の下へ動く = カメラへ近づく


def occupancy(frames_big, ref_rows, cand, motion, dirn):
    """候補車線ごとの「車体と確実に重なる位置に基準車がいたフレーム数」。"""
    S = frames_big[[col("s", f) for f in F_VALUES]].to_numpy()
    # dirn: この車道の進行方向（s が増える向きなら +1）。トラック自身の動きからは決めない
    # （停止中の車両は動きがノイズだけなので、向きが逆になり車体の範囲を反対側に取ってしまう）
    S = S * dirn                                   # 進行方向が正になる向きにそろえる
    e = pd.DataFrame({"Frame": frames_big.Frame.values, "lo": S.min(1), "hi": S.max(1),
                      "tx1": frames_big.BBox_x1.values, "ty1": frames_big.BBox_y1.values,
                      "tx2": frames_big.BBox_x2.values, "ty2": frames_big.BBox_y2.values})
    m = ref_rows[ref_rows.lane.isin(cand)].merge(e, on="Frame")
    iw = (np.minimum(m.BBox_x2, m.tx2) - np.maximum(m.BBox_x1, m.tx1)).clip(lower=0)
    ih = (np.minimum(m.BBox_y2, m.ty2) - np.maximum(m.BBox_y1, m.ty1)).clip(lower=0)
    m = m[iw * ih / ((m.BBox_x2 - m.BBox_x1) * (m.BBox_y2 - m.BBox_y1)) < MAX_INSIDE]
    sc = m[col("s", 0.5)] * dirn
    L, M = TRUCK_LEN_M, MARGIN_M
    if motion == "approach":    # 下辺 = 前端。前端は [lo, hi] のどこか → 車体は前端から後ろへ L
        hit = (sc > m.hi - L + M) & (sc < m.lo - M)
    elif motion == "recede":    # 下辺 = 後端。車体は後端から前へ L
        hit = (sc > m.hi + M) & (sc < m.lo + L - M)
    else:                       # 真横: 下辺が車体全長に掛かる
        hit = (sc > m.lo + M) & (sc < m.hi - M)
    m = m[hit]
    frames = {ln: int(m[m.lane == ln].Frame.nunique()) for ln in cand}
    cars = {ln: int(m[m.lane == ln].Vehicle_ID.nunique()) for ln in cand}
    return frames, cars


def main(cam):
    df = pd.read_csv(out_path(cam, "frenet.csv"))
    df = df[df.s_valid]
    v = per_vehicle(df)
    v = v[(v.n >= MIN_FRAMES) & v.cls.isin([2, 5, 7])].copy()

    # ---- 1. 車線中心と車道 ----
    cen_all = ft.find_lane_centers(v[v.cls == 2].d_med)
    gap = np.diff(cen_all)
    groups = np.split(cen_all, np.where(gap > CARRIAGEWAY_GAP_RATIO * np.median(gap))[0] + 1)
    own = min(groups, key=lambda g: np.abs(g).min())          # 中心線（d=0）を含む車道
    i, _ = nearest(v.d_med, cen_all)
    v = v[np.isin(cen_all[i], own)].copy()                    # 対向車道（別の H が必要）は除外
    print(f"[{cam}] 車線中心 {np.round(cen_all, 2)} → 評価する車道 {np.round(own, 2)}")
    print(f"  車線間隔 {np.round(np.diff(own), 2)} m （正しく換算できていれば約3.6〜3.9 m）")

    # ---- カメラ位置と車線名（カメラに近い順に L1, L2, ...） ----
    cxy, s, tan = ft.fit_centerline(np.load(os.path.join(out_dir(cam), "centerline_f0.500.npy")))
    cam_xy, cam_h, foc = camera_position(load_H(cam), cxy.mean(0))
    _, cam_d = ft.cart_to_frenet(cam_xy[None, :], cxy, s, tan)
    cam_d = float(cam_d[0])
    order = np.argsort(np.abs(own - cam_d))
    lanes = pd.DataFrame({"lane": [f"L{k + 1}" for k in range(len(own))],
                          "center_d": own[order],
                          "lat_from_cam_m": np.abs(own[order] - cam_d),
                          "toward_cam_sign": np.sign(cam_d - own[order])})
    print(f"  カメラ位置 ({cam_xy[0]:.1f}, {cam_xy[1]:.1f}) 高さ {cam_h:.1f} m, 中心線から d={cam_d:+.1f} m")
    if not 25 < cam_h < 40:
        print("  警告: カメラ高さが不自然です。ホモグラフィ／SCALE を確認してください")
    lane_names = dict(zip(lanes.center_d, lanes.lane))
    centers = np.sort(own)
    names = np.array([lane_names[c] for c in centers])

    motion = MOTION or detect_motion(v)
    print(f"  画角: {motion}")

    # ---- 2. クラス（大きさ基準） ----
    i, off = nearest(v.d_med, centers)
    v["lane05"], v["off05"] = names[i], off
    ref_size = v[(v.cls == 2) & (np.abs(v.off05) <= REF_OFF_M)].groupby("lane05")[["w", "h"]].median()
    v["size"] = np.sqrt(v.w / v.lane05.map(ref_size.w) * v.h / v.lane05.map(ref_size.h))
    big_cls = v.cls.isin([5, 7])
    v["group"] = np.select([(v.cls == 2) & (v["size"] < SIZE_CAR_MAX),
                            big_cls & (v["size"] < SIZE_CAR_MAX),
                            big_cls & (v["size"] >= SIZE_LARGE_MIN)],
                           ["car", "pickup", "large"], default="other")

    # ---- 3. 普通車サイズの車両は f=0.5 の最寄り車線 ----
    v["lane"], v["lane_method"], v["conflicts"] = None, None, ""
    small = v.group.isin(["car", "pickup"]) & (np.abs(v.off05) <= LANE_HALF_M)
    v.loc[small, "lane"] = v.loc[small, "lane05"]
    v.loc[small, "lane_method"] = "nearest_f0.5"

    # ---- 4. 大型車は占有で判定 ----
    ref = v[(v["size"] < SIZE_CAR_MAX) & (v.d_shift <= LANE_CHANGE_M) & (np.abs(v.off05) <= REF_OFF_M)]
    ref_rows = df[df.Vehicle_ID.isin(ref.index)][["Frame", "Vehicle_ID", col("s", 0.5),
                                                 "BBox_x1", "BBox_y1", "BBox_x2", "BBox_y2"]]
    ref_rows = ref_rows.assign(lane=ref_rows.Vehicle_ID.map(ref.lane05))
    # 車道の進行方向 = 基準車の s の時間変化の向きの多数決
    v_ref = df[df.Vehicle_ID.isin(ref.index)].groupby("Vehicle_ID")[col("s", 0.5)].agg(lambda x: x.iloc[-1] - x.iloc[0])
    dirn = 1.0 if (v_ref > 0).sum() >= (v_ref < 0).sum() else -1.0
    ref_cen = {f: ref[ref.cls == 2].groupby("lane05")[col("dm", f)].median() for f in (0.0, 0.5, 1.0)}
    big = v[(v.group == "large") & (v.d_shift <= LANE_CHANGE_M)]
    rows_of = {k: g for k, g in df[df.Vehicle_ID.isin(big.index)].groupby("Vehicle_ID")}
    for vid, b in big.iterrows():
        picks = [ref_cen[f].index[np.abs(ref_cen[f].values - b[col("dm", f)]).argmin()] for f in ref_cen]
        idx = sorted(list(names).index(p) for p in picks)
        cand = list(names[idx[0]: idx[-1] + 1])            # f=0〜1 で割り当てられうる車線とその間
        if len(cand) == 1:
            v.loc[vid, ["lane", "lane_method"]] = [cand[0], "unique"]
            continue
        frames, cars = occupancy(rows_of[vid].sort_values("Frame"), ref_rows, cand, motion, dirn)
        srt = sorted(frames, key=frames.get)
        best, others = srt[0], srt[1:]
        v.loc[vid, "conflicts"] = str({ln: f"{frames[ln]}f/{cars[ln]}台" for ln in cand})
        clear = all(frames[o] >= K_CONFLICT and cars[o] >= MIN_CONFLICT_CARS for o in others[:1])
        if frames[best] <= MAX_SELF and clear:
            v.loc[vid, ["lane", "lane_method"]] = [best, "occupancy"]
        else:
            v.loc[vid, "lane_method"] = "undecided"

    v.loc[v.lane_method.isna() & (v.d_shift > LANE_CHANGE_M), "lane_method"] = "lane_change"
    v["regime"] = np.where(v.speed_kmh < CONGEST_KMH, "congested", "free")
    v["lane_center"] = v.lane.map(dict(zip(lanes.lane, lanes.center_d)))
    v["off"] = v.d_med - v.lane_center
    v = v.reset_index()
    v.to_csv(out_path(cam, "vehicles.csv"), index=False, float_format="%.4f")
    lanes["motion"] = motion
    lanes.to_csv(out_path(cam, "lanes.csv"), index=False, float_format="%.3f")

    lg = v[v.group == "large"]
    print(f"  クラス: car {sum(v.group == 'car')}, pickup {sum(v.group == 'pickup')}, "
          f"large {len(lg)}, other {sum(v.group == 'other')}")
    print(f"  大型車の車線判定: {lg.lane_method.value_counts(dropna=False).to_dict()}")
    print(f"  速度（中央値）{v.speed_kmh.median():.0f} km/h, 渋滞中(<{CONGEST_KMH:.0f} km/h)の車両 "
          f"{(v.regime == 'congested').mean():.0%}")


if __name__ == "__main__":
    for cam in (sys.argv[1:] or DEFAULT_CAMERAS):
        main(cam)
