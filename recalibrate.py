# -*- coding: utf-8 -*-
"""
self_calibrate2.py — BEV自己校正 改良版（ペンキの飛ばしOK）

前版からの変更点:
  - ペンキ1本につき「奥端→手前端」の2クリックをワンセット(ペア)とする
  - ペンキは何本飛ばしてもよい。クリック確定後に
    「前のペンキから何本目か」をターミナルで聞くので、目で数えて答える
    (すぐ隣のペンキなら 1、1本飛ばしたら 2、2本飛ばしたら 3)
  - クリック結果を calib_points.npz に保存するので、
    やり直しは `solve` モードで再計算だけできる(クリック不要)

使い方:
  python self_calibrate2.py click    # クリック → スキップ数入力 → 探索
  python self_calibrate2.py solve    # 保存済みクリックで探索だけやり直す
  python self_calibrate2.py apply    # CSVのBEV座標を再計算

クリックの手引き:
  - ラインA: 破線のペンキを2〜3本選び、各ペンキの「奥端→手前端」の順で
    2クリックずつ。ペンキ同士は離れていてよい(奥・中・手前に散らすと良い)
  - [n] でラインBへ。同じ要領で2〜3本
  - [q or Enter] で確定 → スキップ数の質問に答える
  - 奥すぎて端がにじんで見えるペンキは使わない(1〜2pxの誤差が数十cmになる)
"""

import os
import sys
import numpy as np
import pandas as pd

VIDEO_PATH = "rakkabutu_highway.mp4"
CSV_PATH   = "traffic_trajectory_data2.csv"
H_NPY      = "homography.npy"
PTS_NPZ    = "calib_points.npz"

PAINT  = 8.0
GAP    = 12.0
CYCLE  = PAINT + GAP     # 20m
LANE_W = 3.5
KAPPA_MAX = 1.0 / 120.0

# 与えられたκ(曲率)に対して、その曲率の円弧上の座標[m]を返す。
# d_lat は基準線(d_lat=0 の線 = ラインA)からの横方向オフセット。
# 画素は一切使わず、規格値とκだけから机上で「理想の白線」を作る関数。
def world_on_curve(s, kappa, d_lat=0.0):
    s = np.asarray(s, dtype=float)
    if abs(kappa) < 1e-9:
        x = np.zeros_like(s); y = s.copy()
        tx = np.zeros_like(s); ty = np.ones_like(s)
    else:
        x = (1.0 - np.cos(kappa * s)) / kappa
        y = np.sin(kappa * s) / kappa
        tx = np.sin(kappa * s); ty = np.cos(kappa * s)
    nx, ny = -ty, tx
    return np.stack([x + d_lat * nx, y + d_lat * ny], axis=1)


def s_values(n_pairs, skips):
    """ペア数とスキップ数リスト(len = n_pairs-1)から各クリック点の弧長を作る。"""
    s = []
    base = 0.0
    for j in range(n_pairs):
        if j > 0:
            base += CYCLE * skips[j - 1]
        s += [base, base + PAINT]
    return np.array(s)

# world_on_curve()で作った理想の白線[m]と、クリック画素をHで変換した位置[m]の
# ずれの二乗和が最小になるHを、最小二乗法で求める。
# 返り値は (H, 各点の残差[m]の配列)。点が余っているので残差はゼロにならず、
# その大きさが「このκの妥当性」を表す。
def fit_H(img_pts, world_pts):
    import cv2
    H, _ = cv2.findHomography(np.float32(img_pts), np.float32(world_pts), 0)
    if H is None:
        return None, np.array([np.inf])
    proj = cv2.perspectiveTransform(
        np.float32(img_pts).reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, np.linalg.norm(proj - np.float32(world_pts), axis=1)


def solve(ptsA, skipsA, ptsB, skipsB):
    """κ, φ, sgn を総当たりして、残差最小の組み合わせとそのHを返す。
    κは計算で求めるのではなく、候補を並べて残差で採点し選ぶ。
    残差が採点に使えるのは、点が余っている(自由度8 < 方程式2N)ため。
    """
    sA = s_values(len(ptsA) // 2, skipsA)
    sB = s_values(len(ptsB) // 2, skipsB) if len(ptsB) else None
    img = np.vstack([ptsA, ptsB]) if len(ptsB) else np.array(ptsA, float)

    best = dict(rms=np.inf)

    def trial(kappa, phi, sgn, keepH=False):
        wA = world_on_curve(sA, kappa, 0.0)
        if len(ptsB):
            wB = world_on_curve(sB + phi, kappa, sgn * LANE_W)
            world = np.vstack([wA, wB])
        else:
            world = wA
        H, err = fit_H(img, world)
        rms = float(np.sqrt(np.mean(err ** 2)))
        if rms < best["rms"]:
            best.update(rms=rms, kappa=kappa, phi=phi, sgn=sgn)
            if keepH:
                best.update(H=H, err=err)
        return rms

    print("粗探索中...")
    for kappa in np.linspace(-KAPPA_MAX, KAPPA_MAX, 121):
        for sgn in ([+1.0, -1.0] if len(ptsB) else [+1.0]):
            for phi in (np.arange(0.0, CYCLE, 0.5) if len(ptsB) else [0.0]):
                trial(kappa, phi, sgn)

    print("微調整中...")
    k0, p0, sg = best["kappa"], best["phi"], best["sgn"]
    for kappa in np.linspace(k0 - 0.0015, k0 + 0.0015, 121):
        for phi in (np.arange(p0 - 0.6, p0 + 0.6, 0.02) if len(ptsB) else [0.0]):
            trial(kappa, phi, sg, keepH=True)
    if "H" not in best:
        trial(best["kappa"], best["phi"], best["sgn"], keepH=True)
    return best


def report(best, nA, nB):
    R = np.inf if abs(best["kappa"]) < 1e-9 else 1.0 / best["kappa"]
    rep = [f"推定曲率 κ = {best['kappa']:.5f} [1/m]  →  半径 R = {R:.0f} m",
           f"ラインB: 位相 φ = {best['phi']:.2f} m, 横位置 = {best['sgn'] * LANE_W:+.1f} m",
           f"当てはめRMS誤差 = {best['rms']:.3f} m",
           "--- 各点の誤差 ---"]
    labels = [f"A{i}" for i in range(nA)] + [f"B{i}" for i in range(nB)]
    for lb, e in zip(labels, best["err"]):
        rep.append(f"  {lb}: {e:.3f} m" + ("  <-- 要確認!" if e > 0.4 else ""))
    text = "\n".join(rep)
    print("\n" + text)
    np.save(H_NPY, best["H"])
    with open("self_calib_report.txt", "w") as f:
        f.write(text + "\n")
    print(f"\n保存: {H_NPY} / self_calib_report.txt")
    print("目安: RMS 0.3m以下で合格 → bev_preview.py で目視確認へ")


def ask_skips(name, n_pairs):
    skips = []
    for j in range(1, n_pairs):
        while True:
            ans = input(f"  {name}: ペンキ{j+1}本目は、その前のペンキから"
                        f"数えて何本目？(隣=1, 1本飛ばし=2, ...): ").strip()
            try:
                v = int(ans)
                if v >= 1:
                    skips.append(v)
                    break
            except ValueError:
                pass
            print("    1以上の整数で入力してください")
    return skips


def click_mode():
    import cv2
    cap = cv2.VideoCapture(VIDEO_PATH)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"動画を読めません: {VIDEO_PATH}")

    lines = [[], []]
    cur = [0]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            lines[cur[0]].append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            cur[0] = 1

    win = "pairs: far-end then near-end per dash | rclick/n=lineB u=undo q=done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    colors = [(0, 0, 255), (255, 120, 0)]

    while True:
        disp = frame.copy()
        for li, pts in enumerate(lines):
            for i, p in enumerate(pts):
                cv2.circle(disp, p, 4, colors[li], -1)
                cv2.putText(disp, f"{'AB'[li]}{i}", (p[0] + 5, p[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[li], 1)
        cv2.putText(disp, f"now: line {'AB'[cur[0]]}  pts A={len(lines[0])} B={len(lines[1])}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("u"), ord("U")) and lines[cur[0]]:
            lines[cur[0]].pop()
        elif k in (ord("n"), ord("N")):
            cur[0] = 1
        elif k in (ord("q"), ord("Q"), 13):
            break
    cv2.destroyAllWindows()

    ptsA, ptsB = lines
    for name, pts in (("A", ptsA), ("B", ptsB)):
        if len(pts) % 2 != 0:
            raise RuntimeError(f"ライン{name}のクリック数が奇数です({len(pts)}点)。"
                               "1本のペンキにつき奥端・手前端の2点セットで")
    if len(ptsA) < 4 or len(ptsB) < 4:
        raise RuntimeError("各ライン最低ペンキ2本(4点)必要です")

    print("\nクリック確認: A={}ペンキ, B={}ペンキ".format(len(ptsA)//2, len(ptsB)//2))
    skipsA = ask_skips("ラインA", len(ptsA) // 2)
    skipsB = ask_skips("ラインB", len(ptsB) // 2)

    np.savez(PTS_NPZ, ptsA=np.array(ptsA, float), ptsB=np.array(ptsB, float),
             skipsA=np.array(skipsA), skipsB=np.array(skipsB))
    print(f"クリックを保存: {PTS_NPZ}(次回は solve モードで再計算だけ可能)")

    best = solve(np.array(ptsA, float), skipsA, np.array(ptsB, float), skipsB)
    report(best, len(ptsA), len(ptsB))


def solve_mode():
    z = np.load(PTS_NPZ)
    best = solve(z["ptsA"], list(z["skipsA"]), z["ptsB"], list(z["skipsB"]))
    report(best, len(z["ptsA"]), len(z["ptsB"]))


def apply_mode():
    import cv2
    H = np.load(H_NPY)
    df = pd.read_csv(CSV_PATH)
    px = df[["X_pixel", "Y_pixel"]].to_numpy(np.float32).reshape(-1, 1, 2)
    mt = cv2.perspectiveTransform(px, H).reshape(-1, 2)
    df["X_meter"], df["Y_meter"] = mt[:, 0], mt[:, 1]
    out = os.path.splitext(CSV_PATH)[0] + "_recal.csv"
    df.to_csv(out, index=False)
    print(f"保存しました: {out} ({len(df)}行)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "click":
        click_mode()
    elif mode == "solve":
        solve_mode()
    elif mode == "apply":
        apply_mode()
    else:
        print(__doc__)