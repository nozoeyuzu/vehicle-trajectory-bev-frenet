# -*- coding: utf-8 -*-
"""
evaluate_refpoint.py — bbox基準点（下辺上の位置 f）ごとのクラス間差・ノイズを集計する

使い方:
    python evaluate_refpoint.py

入力: frenet_transform.py convert の出力（*_frenet_auto.csv）を f ごとに1本ずつ
出力:
    refpoint_summary.csv   車線 × f ごとの集計表（主表）
    refpoint_vehicles.csv  車両ごとの代表値（信頼区間の元データ）
    refpoint_sweep.png     横軸 f・縦軸クラス間差の折れ線（車線ごと）

評価指標（進捗報告の定義に対応）:
    基準    : 各車線で、普通車（多数決 Class_ID=2, 20フレーム以上）の d 中央値を車線中心とし
              d_dev = d − 車線中心
    クラス   : 車両単位の多数決。car=普通車、truck+bus=大型車
    代表値   : 車両ごとの d_dev の平均
    主指標   : 大型車の代表値の平均 − 普通車の代表値の平均（符号付き、Welch 95%区間、Cohen's d）
    副指標   : 車両内 d_dev 標準偏差の中央値（軌跡の綺麗さ）
    入口条件 : 20フレーム以上、車線変更（追跡前半と後半の d 中央値の差 > 2.0 m）と
              車線またぎ（車両中央値が車線中心から 1.0 m 超）は除外
    車線割当と除外はデフォルト（f=0.5）で決め、全 f で同じ車両集合を使う
"""

import os
import numpy as np
import pandas as pd

# ============== 設定 ==============
BASE_NAME  = "scene_A_track2_recal2"     # make_refpoint_csv.py に渡した CSV の名前（拡張子なし）
OUT_DIR    = "out_" + BASE_NAME        # make_refpoint_csv.py と同じ規則。入力も出力もここ
F_VALUES   = [i / 8 for i in range(9)]   # 0, 0.125, ..., 1 の対称な9点
INPUTS     = {f: os.path.join(OUT_DIR, f"{BASE_NAME}_f{f:.3f}_frenet_auto.csv") for f in F_VALUES}
BASELINE_F = 0.5                       # 車線割り当て・除外を決める基準（下辺中央）
MAIN_F     = [0.0, 0.5, 1.0]           # 主評価の固定3点（表で印を付ける）

#ここ画角によって変える必要あり
LANE_BOUNDS = [-3.5, 1.5, 6.0]         # 車両中央値 d をこの境界で車線に分ける（d の小さい順）
LANE_NAMES  = ["L1", "L2", "L3", "L4"] # 境界数+1 個
EVAL_LANES  = ["L2", "L3"]             # 集計する車線。None で全車線

CAR_CLASSES   = {2}                    # 普通車
LARGE_CLASSES = {5, 7}                 # 大型車（bus + truck）
MIN_FRAMES    = 20                     # これ未満の車両は除外
LANE_CHANGE_M = 2.0                    # 追跡の前半1/4と後半1/4の d 中央値の差がこれを超える車両は車線変更として除外
                                       # （隣の車線中心まで約3.5 m。max−min だと近距離の大型車のぐらつきを誤検出する）
STRADDLE_M    = 1.0                    # 車両中央値が車線中心からこれ以上離れていれば車線またぎとして除外
                                       # （車線半幅 1.75 m − 普通車半幅 0.9 m ≈ 0.85 m を超えると車体が白線に掛かる）
MIN_LARGE     = 2                      # 大型車がこれ未満の車線はクラス間差を出さない
S_BINS        = [0, 20, 40, 60, 80, 100]  # s帯別の補助表の境界 [m]
MAKE_PLOT     = True
# ==================================


def _shift(d):
    """追跡の前半1/4と後半1/4の d 中央値の差（車線変更なら車線幅程度になる）。"""
    q = max(3, len(d) // 4)
    return abs(np.median(d[-q:]) - np.median(d[:q]))


def per_vehicle(df):
    """車両ごとの多数決クラス・d 中央値・d の変動幅・フレーム数。"""
    g = df.groupby("Vehicle_ID")
    v = pd.DataFrame({
        "cls":     g["Class_ID"].agg(lambda s: s.mode().iloc[0]),
        "d_med":   g["d_meter"].median(),
        "d_range": g["d_meter"].max() - g["d_meter"].min(),
        "d_shift": g.apply(lambda x: _shift(x.sort_values("Frame")["d_meter"].to_numpy())),
        "n":       g.size(),
    })
    v["group"] = np.where(v.cls.isin(CAR_CLASSES), "car",
                 np.where(v.cls.isin(LARGE_CLASSES), "large", "other"))
    return v


def assign_lane(d_med):
    return np.array(LANE_NAMES)[np.digitize(d_med, LANE_BOUNDS)]


def show_lane_overview(df, veh):
    """LANE_BOUNDS / EVAL_LANES を決めるための一覧。
    車両ごとの d 中央値の分布と、車線ごとの台数・画像上の位置・進行方向を表示する。"""
    print("\n=== 車両ごとの d 中央値の分布（山と山の間に LANE_BOUNDS を置く） ===")
    h, b = np.histogram(veh.d_med, bins=np.arange(np.floor(veh.d_med.min()), np.ceil(veh.d_med.max()) + 0.5, 0.5))
    for c, lo, hi in zip(h, b[:-1], b[1:]):
        if c:
            print(f"  {lo:+6.1f} 〜 {hi:+6.1f} m | {'#' * c}")
    print("\n=== 車線ごとの概要（EVAL_LANES を決める。手前=Y_pixel が大きい） ===")
    df = df[df.Vehicle_ID.isin(veh.index)].copy()
    df["lane"] = df.Vehicle_ID.map(veh.lane)
    dy = df.sort_values("Frame").groupby("Vehicle_ID")["Y_meter"].agg(lambda s: s.iloc[-1] - s.iloc[0])
    rows = []
    for lane, g in df.groupby("lane"):
        vl = veh[veh.lane == lane]
        near = g.nlargest(max(20, len(g) // 10), "Y_pixel")     # 画面手前側の点
        rows.append(dict(lane=lane, n_car=(vl.group == "car").sum(), n_large=(vl.group == "large").sum(),
                         d_med=round(vl.d_med.median(), 2),
                         Y_pixel_max=round(g.Y_pixel.max()), X_pixel_near=round(near.X_pixel.median()),
                         bbox_h_near=round((near.BBox_y2 - near.BBox_y1).median()),
                         dir=("+" if dy[vl.index].median() > 0 else "-")))
    print(pd.DataFrame(rows).to_string(index=False))
    print("  Y_pixel_max が画像下端に近い車線ほど手前。dir が異なる車線は対向。n_large<2 の車線は評価不可。\n")


def decide_vehicle_set(path):
    """BASELINE_F の結果から、車線割り当てと入口条件を満たす車両集合を決める。
    戻り値: (lane_map: Vehicle_ID→lane, excluded_lane_change: Vehicle_ID→lane) """
    df = pd.read_csv(path)
    df = df[df["s_valid"]]
    veh = per_vehicle(df)
    veh = veh[(veh.n >= MIN_FRAMES) & (veh.group != "other")].copy()
    veh["lane"] = assign_lane(veh.d_med.to_numpy())
    show_lane_overview(df, veh)
    if EVAL_LANES is not None:
        veh = veh[veh.lane.isin(EVAL_LANES)]
    centers = veh[veh.group == "car"].groupby("lane")["d_med"].median()
    off = (veh.d_med - veh.lane.map(centers)).abs()
    bad = (veh.d_shift > LANE_CHANGE_M) | (off > STRADDLE_M)
    lc, keep = veh[bad], veh[~bad]
    print(f"車両集合（f={BASELINE_F} で決定）: {len(veh)}台中 車線変更/またぎとして除外 {len(lc)}台 → {len(keep)}台")
    if len(lc):
        print(lc[["lane", "group", "n", "d_med", "d_shift", "d_range"]].round(2).to_string())
    return keep[["lane", "group"]], lc[["lane", "group"]]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (b.mean() - a.mean()) / sp if sp > 0 else np.nan


def summarize(veh, label):
    """車線ごとに主指標・副指標を集計する。veh は d_dev 代表値を持つ車両表。"""
    rows = []
    for lane, vl in veh.groupby("lane"):
        car = vl[vl.group == "car"]
        big = vl[vl.group == "large"]
        row = dict(f=label, lane=lane, n_car=len(car), n_large=len(big),
                   noise_sd_cm=round(vl.d_dev_sd.median() * 100, 1),
                   noise_dd_cm=round(vl.dd_sd.median() * 100, 1))
        if len(big) >= MIN_LARGE and len(car) >= 2:
            diff = big.d_dev_mean.mean() - car.d_dev_mean.mean()
            se = np.sqrt(car.d_dev_mean.var(ddof=1) / len(car) +
                         big.d_dev_mean.var(ddof=1) / len(big))
            row.update(class_diff_m=round(diff, 3),
                       ci95_m=round(1.96 * se, 3),
                       cohens_d=round(cohens_d(car.d_dev_mean, big.d_dev_mean), 2),
                       median_diff_m=round(big.d_dev_mean.median() - car.d_dev_mean.median(), 3),
                       car_sd_between_cm=round(car.d_dev_mean.std() * 100, 1),
                       large_sd_between_cm=round(big.d_dev_mean.std() * 100, 1))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_one(f, path, lane_map):
    """1つの f について、指定された車両集合で d_dev を作り直し、集計する。"""
    df = pd.read_csv(path)
    df = df[df["s_valid"] & df.Vehicle_ID.isin(lane_map.index)].copy()
    veh = per_vehicle(df)
    veh["lane"]  = veh.index.map(lane_map["lane"])
    veh["group"] = veh.index.map(lane_map["group"])   # クラスも BASELINE_F で確定した値を使う

    # 車線中心 = 普通車の車両中央値の中央値（この f で作り直す）
    centers = veh[veh.group == "car"].groupby("lane")["d_med"].median()
    veh["lane_center"] = veh.lane.map(centers)
    df["lane_center"] = df.Vehicle_ID.map(veh.lane_center)
    df["d_dev"] = df["d_meter"] - df["lane_center"]
    df = df.sort_values(["Vehicle_ID", "Frame"])
    df["dd"] = df.groupby("Vehicle_ID")["d_dev"].diff()

    g = df.groupby("Vehicle_ID")
    veh["d_dev_mean"] = g["d_dev"].mean()     # 代表値
    veh["d_dev_sd"]   = g["d_dev"].std()
    veh["dd_sd"]      = g["dd"].std()
    veh["f"] = f

    # s帯別の補助表
    df["group"] = df.Vehicle_ID.map(veh.group)
    df["lane"]  = df.Vehicle_ID.map(veh.lane)
    df["sbin"]  = pd.cut(df.s_meter, S_BINS)
    sb = df.pivot_table(index=["lane", "sbin"], columns="group", values="d_dev",
                        aggfunc="mean", observed=True)
    sband = None
    if {"car", "large"} <= set(sb.columns):
        sband = (sb["large"] - sb["car"]).unstack("sbin").round(3)
        sband.insert(0, "f", f)
    return summarize(veh, f), veh.reset_index(), sband


def main():
    lane_map, lane_change = decide_vehicle_set(INPUTS[BASELINE_F])
    lane_map_all = pd.concat([lane_map, lane_change])   # 除外なし（感度確認用）
    lane_map, lane_map_all = lane_map.copy(), lane_map_all.copy()

    summaries, vehicles, sbands = [], [], []
    for f, path in INPUTS.items():
        s, v, sb = evaluate_one(f, path, lane_map)
        summaries.append(s); vehicles.append(v)
        if sb is not None:
            sbands.append(sb)
        # 除外なしの主指標も一列添える
        s_all, _, _ = evaluate_one(f, path, lane_map_all)
        s = s.merge(s_all[["f", "lane", "class_diff_m"]].rename(columns={"class_diff_m": "class_diff_noexcl_m"}),
                    on=["f", "lane"], how="left")
        summaries[-1] = s

    summary = pd.concat(summaries, ignore_index=True)
    summary["main"] = summary.f.isin(MAIN_F).map({True: "*", False: ""})
    cols = ["lane", "f", "main", "n_car", "n_large", "class_diff_m", "ci95_m", "cohens_d",
            "median_diff_m", "class_diff_noexcl_m", "noise_sd_cm", "noise_dd_cm",
            "car_sd_between_cm", "large_sd_between_cm"]
    summary = summary[[c for c in cols if c in summary.columns]]
    summary = summary.sort_values(["lane", "f"]).reset_index(drop=True)
    summary.to_csv(os.path.join(OUT_DIR, "refpoint_summary.csv"), index=False)
    pd.concat(vehicles, ignore_index=True).to_csv(os.path.join(OUT_DIR, "refpoint_vehicles.csv"), index=False)

    pd.set_option("display.width", 250)
    print("\n=== 車線 × f の集計（* = 主評価の固定3点） ===")
    print(summary.to_string(index=False))
    if sbands:
        print("\n=== s帯別 クラス間差 [m]（フレーム平均、距離依存性の確認用） ===")
        print(pd.concat(sbands).sort_index().to_string())
    print("""
読み方:
  class_diff_m        : 大型車 − 普通車 の代表値平均の差 [m]（主指標、0に近いほど良い）
  ci95_m              : その差の95%区間の半幅 [m]（Welch）
  cohens_d            : 効果量（先行研究 highD の同一車線内の車種差は < 0.35）
  median_diff_m       : 中央値の差 [m]（外れ値への頑健性確認）
  class_diff_noexcl_m : 車線変更車を除外しなかった場合の主指標（感度確認）
  noise_sd_cm         : 車両内 d_dev の標準偏差の中央値 [cm]（副指標、小さいほど良い）
  noise_dd_cm         : 車両内フレーム間 Δd の標準偏差の中央値 [cm]
  *_sd_between_cm     : 代表値のクラス内ばらつき [cm]（区間幅の元）""")

    if MAKE_PLOT:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lane, sl in summary.dropna(subset=["class_diff_m"]).groupby("lane"):
            ax.errorbar(sl.f, sl.class_diff_m, yerr=sl.ci95_m, marker="o", capsize=3, label=lane)
        ax.axhline(0, color="0.6", lw=1)
        for f0 in MAIN_F:
            ax.axvline(f0, color="0.85", lw=1, ls="--")
        ax.set_xlabel("f  (0 = bottom-left, 0.5 = bottom-center, 1 = bottom-right)")
        ax.set_ylabel("class difference  large - car  [m]")
        ax.set_title("Class difference vs reference-point position (95% CI)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "refpoint_sweep.png"), dpi=130)
        print("図を保存: refpoint_sweep.png")


if __name__ == "__main__":
    main()