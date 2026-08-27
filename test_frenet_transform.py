# # -*- coding: utf-8 -*-
# """test_frenet.py — Frenet変換(frenet.py)の検証スイート

# ============================================================================
#  このテストが検証すること・しないこと
# ============================================================================
# 検証する : 中心線を「与えられたもの」としたときの変換の正しさ
# 検証しない: 中心線そのものが道路に沿っているか

#   中心線を30度傾けても、その傾いた線で測って同じ線で戻せば往復は成立する。
#   つまり本ファイルの検査は中心線の正しさから完全に独立している。
#   中心線の妥当性は「車線維持車の d(s) が水平か」で別途確認すること
#   (対向車線22本で -0.10 m/100m を確認済み)。

# ============================================================================
#  構成
# ============================================================================
#  層A 実装の正しさ   合成データ。解析解と比較する
#    A1 test_straight_exact              直線: s=y, d=-x が厳密に出るか
#    A2 test_circular_known_offsets      円弧: 既知の法線オフセットを復元するか
#    A3 test_off_sample_query            格納点と一致しない点で検証(空テスト対策)
#    A4 test_round_trip_synthetic        (s,d)->(X,Y) の往復
#    A5 test_sampling_stability          サンプル数を変えても答えが変わらないか
#    A6 test_fit_centerline_straight     fit_centerline が直線を直線に写すか
#    A7 test_fit_centerline_duplicate    重複クリック点への耐性

#  層B 定義域の境界   外挿の挙動と誤差の予測式
#    B1 test_beyond_ends_is_linear       中心線の外は末端接線の直線になること
#    B2 test_extrapolation_error_law     誤差が ext^2/(2R) に従うこと

#  層C 実データ       層Bの式を使って許容範囲を判定
#    C1 report_centerline_quality        クリック点の質(残差・重複・被覆)
#    C2 assert_round_trip_real           往復誤差
#    C3 assert_perpendicular_foot        垂線の足が取れているか
#    C4 report_reverse_steps             進行方向を考慮した逆行率
#    C5 assert_coverage                  外挿量から有効な s 上限を算出

# ============================================================================
#  使い方
# ============================================================================
#     python test_frenet.py                 # 層A+B のみ(高速・環境非依存)
#     python test_frenet.py --real          # 層Cも実行(診断を表示)
#     python test_frenet.py --real --strict # 層Cの違反で異常終了(CI用)
#     python test_frenet.py --real --max-extrap-error 0.20
#                                           # 許容する外挿誤差[m]を指定
# """
# import argparse
# import sys

# import numpy as np
# import pandas as pd
# from scipy.spatial import cKDTree

# import frenet as ft

# ROAD_RADIUS = 1100.0   # 実道路の曲率半径[m]。較正 1101 m / 画像計測 960 m
# _failures = []


# # ============================================================
# # 補助
# # ============================================================
# def check(name, actual, expected, tol, note=""):
#     err = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
#     ok = err <= tol
#     tag = "OK  " if ok else "FAIL"
#     print(f"  {tag} {name}: max err = {err:.4g} (tol {tol:.4g}) {note}")
#     if not ok:
#         _failures.append(name)
#     return ok


# def circular_centerline(radius=50.0, a0=0.0, a1=np.pi / 2, samples=20001):
#     """解析解が既知の円弧中心線。s = R*theta, 法線オフセット = d。"""
#     th = np.linspace(a0, a1, samples)
#     cxy = radius * np.column_stack([np.cos(th), np.sin(th)])
#     s = radius * (th - a0)
#     tan = np.column_stack([-np.sin(th), np.cos(th)])
#     return cxy, s, tan, th


# def left_normal(tan):
#     """接線に対する左手法線。d の正方向と一致する。"""
#     return np.column_stack([-tan[:, 1], tan[:, 0]])


# def frenet_to_cart(s_out, d_out, cxy, s, tan):
#     """(s,d) -> (X,Y) の逆変換.

#     順変換は KD木で足を選ぶが、逆変換では s しか手元に無いので
#     searchsorted で足を引き直す。この非対称性が往復誤差の主因になる。
#     """
#     idx = np.clip(np.searchsorted(s, s_out), 0, len(s) - 1)
#     tg = tan[idx]
#     return cxy[idx] + (s_out - s[idx])[:, None] * tg + d_out[:, None] * left_normal(tg)


# def round_trip_tolerance(d_out, s, radius):
#     """往復誤差の許容値を誤差機構から導出する.

#     逆変換は足を最大1サンプルずらす。曲線上ではそのずれが
#     |d| に比例した誤差 (|d| * spacing / R) を生む。その10倍を上限とする。
#     「たぶんこのくらい」で決めないこと。値が動いたとき、実装が変わったのか
#     離散化が変わったのかを区別できなくなる。
#     """
#     spacing = s[-1] / len(s)
#     return max(10.0 * float(np.abs(d_out).max()) * spacing / radius, 1e-9)


# # ============================================================
# # 層A  実装の正しさ
# # ============================================================
# def test_straight_exact():
#     """A1 直線中心線 c(s)=(0,s) では s=y, d=-x が厳密に成立する.

#     得られるもの: 符号の規約が「進行方向に向かって左が正」であることの確認。
#       接線 (0,1) の左は -x 側なので d=-x。将来ここが反転したら
#       LANE_CENTERS_D の並びが全部裏返るので、真っ先に落ちてほしい検査。
#     """
#     print("\n[A1] 直線中心線での厳密解")
#     y = np.linspace(0.0, 100.0, 10001)
#     cxy = np.column_stack([np.zeros_like(y), y])
#     tan = np.tile([0.0, 1.0], (len(y), 1))
#     pts = np.array([[0., 5.], [2., 10.], [-3., 25.], [1.5, 70.25]])
#     a_s, a_d = ft.cart_to_frenet(pts, cxy, y.copy(), tan)
#     check("s = y", a_s, pts[:, 1], 1e-10)
#     check("d = -x (左が正)", a_d, -pts[:, 0], 1e-10)


# def test_circular_known_offsets():
#     """A2 円弧中心線で、既知の法線オフセットを復元できるか.

#     得られるもの: 曲率がある場合の s(弧長)と d(符号付き横距離)の正しさ。
#       実道路(R=1100 m)より遥かに急な R=50 m で試すことで、
#       曲率に起因する誤差を意図的に増幅した状態で検証している。
#     """
#     print("\n[A2] 円弧中心線での既知オフセット復元")
#     R = 50.0
#     cxy, s, tan, _ = circular_centerline(radius=R)
#     th = np.array([0.15, 0.45, 0.9, 1.25])
#     base = R * np.column_stack([np.cos(th), np.sin(th)])
#     tg = np.column_stack([-np.sin(th), np.cos(th)])
#     off = np.array([-3.0, 0.0, 1.5, 4.0])
#     pts = base + off[:, None] * left_normal(tg)
#     a_s, a_d = ft.cart_to_frenet(pts, cxy, s, tan)
#     spacing = s[-1] / len(s)
#     check("s = R*theta", a_s, R * th, 3 * spacing, f"(間隔 {spacing*1000:.2f} mm)")
#     check("d = 既知オフセット", a_d, off, 3 * spacing)


# def test_off_sample_query():
#     """A3 中心線サンプルと一致しない点で検証する.

#     なぜ必要か: cart_to_frenet(cxy, cxy, ...) のようにサンプル点そのものを
#       問い合わせると rel=0 が厳密に成立し、d=0 は計算せずとも決まる。
#       「誤差が厳密に 0」というテストは通ったのではなく、何も試していない。
#       ここではサンプル間の中点を使い、最近傍探索と接線補正を実際に働かせる。
#     """
#     print("\n[A3] サンプル点と一致しない点での検証(空テスト対策)")
#     R = 50.0
#     cxy, s, tan, _ = circular_centerline(radius=R, samples=5001)
#     mid = 0.5 * (cxy[:-1] + cxy[1:])          # 弦の中点。円の内側に僅かに入る
#     _, d_mid = ft.cart_to_frenet(mid, cxy, s, tan)
#     sag = R * (1 - np.cos(np.pi / 2 / 5000 / 2))   # 弦の凹み量(理論値)
#     print(f"       中点の |d| 最大 = {np.abs(d_mid).max():.3e} m "
#           f"(弦の凹み理論値 {sag:.3e} m)")
#     off = 1.234
#     pts = mid + off * left_normal(tan[:-1])
#     _, d_off = ft.cart_to_frenet(pts, cxy, s, tan)
#     check("既知オフセット 1.234 m の復元", d_off, np.full(len(pts), off), 5e-3)


# def test_round_trip_synthetic():
#     """A4 (s,d) -> (X,Y) の往復.

#     得られるもの: 最も強い単一の主張 ——
#       「(s,d) は (X,Y) と同じ情報を持つ座標系の付け替えに過ぎない」。
#       往復が成立するなら、変換の過程で情報は一切失われていない。
#       ただし中心線が正しい場所にあるかは何も言っていないことに注意。
#     """
#     print("\n[A4] 逆変換の往復(合成)")
#     R = 50.0
#     cxy, s, tan, _ = circular_centerline(radius=R)
#     th = np.linspace(0.05, 1.5, 200)
#     base = R * np.column_stack([np.cos(th), np.sin(th)])
#     tg = np.column_stack([-np.sin(th), np.cos(th)])
#     off = np.linspace(-4, 4, 200)
#     xy = base + off[:, None] * left_normal(tg)
#     s_out, d_out = ft.cart_to_frenet(xy, cxy, s, tan)
#     rec = frenet_to_cart(s_out, d_out, cxy, s, tan)
#     err = np.linalg.norm(rec - xy, axis=1)
#     tol = round_trip_tolerance(d_out, s, R)
#     check("往復誤差", err, np.zeros(len(err)), tol,
#           f"(|d|max {np.abs(d_out).max():.1f} m)")


# def test_sampling_stability():
#     """A5 中心線のサンプル数を変えても答えが変わらないか.

#     注意: この検査は外挿の誤りに対しては盲目。中心線の外側の点では
#       どのサンプル数でも同じ末端接線が使われるため、2つの誤った答えが
#       一致して合格してしまう。境界の検査は層Bが担当する。
#     """
#     print("\n[A5] サンプリング安定性")
#     R = 50.0
#     th = np.array([0.2, 0.6, 1.0, 1.35])
#     base = R * np.column_stack([np.cos(th), np.sin(th)])
#     tg = np.column_stack([-np.sin(th), np.cos(th)])
#     pts = base + np.array([2., -1., 3., -2.5])[:, None] * left_normal(tg)
#     c = circular_centerline(radius=R, samples=3001)[:3]
#     d = circular_centerline(radius=R, samples=30001)[:3]
#     cr = ft.cart_to_frenet(pts, *c)
#     dr = ft.cart_to_frenet(pts, *d)
#     check("s の一致 (3001 vs 30001)", cr[0], dr[0], 2e-3)
#     check("d の一致 (3001 vs 30001)", cr[1], dr[1], 2e-3)


# def test_fit_centerline_straight():
#     """A6 fit_centerline が直線クリック点から直線を作るか.

#     得られるもの: 多項式フィット部分の健全性。既存テストは
#       cart_to_frenet だけを試し、fit_centerline には assert が無かった。
#     """
#     print("\n[A6] fit_centerline: 直線の再現")
#     pts = np.array([[0., 0.], [0., 25.], [0., 50.], [0., 75.], [0., 100.]])
#     cxy, s, tan = ft.fit_centerline(pts, n=2001)
#     check("横方向のふらつき", cxy[:, 0], np.zeros(len(cxy)), 1e-6)
#     check("全長 = 100 m", [s[-1]], [100.0], 1e-6)
#     check("接線 = (0,1)", tan, np.tile([0., 1.], (len(cxy), 1)), 1e-6)
#     if np.any(np.diff(s) <= 0):
#         _failures.append("s monotonic")
#         print("  FAIL s が単調増加していません")
#     else:
#         print("  OK   s は単調増加")


# def test_fit_centerline_duplicate():
#     """A7 重複クリック点への耐性.

#     なぜ必要か: 実データの centerline_meter.npy には 0.131 m しか
#       離れていない2点がある(ダブルクリックの取りこぼし)。弦長パラメータ上で
#       ほぼ同じ位置に2つの拘束が入り、末端が二重に重み付けされる。
#     """
#     print("\n[A7] fit_centerline: 重複クリック点への耐性")
#     clean = np.array([[0., 0.], [0., 25.], [0., 50.], [0., 75.], [0., 100.]])
#     dup = np.vstack([clean, clean[-1] + [0.05, 0.10]])
#     c1, s1, _ = ft.fit_centerline(clean, n=2001)
#     c2, s2, _ = ft.fit_centerline(dup, n=2001)
#     dev = float(np.abs(c2[:, 0]).max())
#     print(f"       横方向の歪み {dev:.4f} m / 全長差 {abs(s2[-1]-s1[-1]):.4f} m")
#     check("重複による歪み", [dev], [0.0], 0.15)


# # ============================================================
# # 層B  定義域の境界
# # ============================================================
# def test_beyond_ends_is_linear():
#     """B1 中心線の外側では末端接線の直線が基準線になることを明示する.

#     なぜ必要か: cart_to_frenet は KD木で最近傍サンプルを引く。中心線より
#       外の点にとって最近傍は必ず末端の1点になり、そこで idx が固定される。
#       すると tg も末端接線に固定され、s_out = s[-1] + rel·tg は
#       「末端接線に沿った直線」を基準にした値になる。
#       既存テストは問い合わせ点を全て中心線の内側に置いていたため、
#       この挙動を一度も踏んでいなかった。
#     """
#     print("\n[B1] 中心線の外側は直線外挿になる")
#     R = 50.0
#     cxy, s, tan, th = circular_centerline(radius=R)
#     end_tan = tan[-1]
#     for ext in [1.0, 3.0, 5.0]:
#         pt = (cxy[-1] + ext * end_tan)[None, :]   # 末端接線上の点
#         s_out, d_out = ft.cart_to_frenet(pt, cxy, s, tan)
#         # 末端接線上なので、直線外挿なら d は厳密に 0、s は s[-1]+ext
#         check(f"接線上 ext={ext:.0f} m の d", d_out, [0.0], 1e-6)
#         check(f"接線上 ext={ext:.0f} m の s", s_out, [s[-1] + ext], 1e-6)


# def test_extrapolation_error_law():
#     """B2 外挿誤差が ext^2/(2R) に従うことを確認する.

#     得られるもの: 実データで「どこまで外挿を許すか」を計算で決める根拠。
#       層Cの assert_coverage はこの式を使って有効な s 上限を出すので、
#       式そのものをここで検証しておかないと判断の根拠が閉じない。

#       検査の作り方: 円弧を延長した先の点は「本来の道路上」にあるので
#       真の d は 0。それを直線外挿した基準線で測った値が、そのまま外挿誤差。
#     """
#     print("\n[B2] 外挿誤差の法則 ext^2/(2R)")
#     for R in [50.0, 300.0]:
#         cxy, s, tan, th = circular_centerline(radius=R, a1=np.pi / 3)
#         print(f"       R = {R:.0f} m")
#         for ext in [2.0, 5.0, 10.0]:
#             th_far = th[-1] + ext / R
#             pt = R * np.array([[np.cos(th_far), np.sin(th_far)]])
#             _, d_out = ft.cart_to_frenet(pt, cxy, s, tan)
#             pred = ext ** 2 / (2 * R)
#             got = abs(float(d_out[0]))
#             rel = abs(got - pred) / pred
#             tag = "OK  " if rel < 0.10 else "FAIL"
#             if rel >= 0.10:
#                 _failures.append(f"extrap law R={R} ext={ext}")
#             print(f"  {tag} ext={ext:5.1f} m: 実測 |d| = {got:.4f} m / "
#                   f"予測 {pred:.4f} m (差 {rel:.1%})")


# # ============================================================
# # 層C  実データ
# # ============================================================
# def report_centerline_quality(raw, cxy, s, xy):
#     """C1 クリック点の質と被覆を報告する.

#     得られるもの:
#       - 残差: 特定の点だけ残差が大きければ、クリックミスか重複を示す
#       - 最小間隔: 1 m を切っていればダブルクリックの疑い
#       - 被覆: 中心線がデータ範囲を覆っているか。ここが外挿問題の根本原因
#     """
#     print("\n[C1] 中心線の質")
#     p = np.asarray(raw, float)
#     seg = np.hypot(*np.diff(p, axis=0).T)
#     t = np.concatenate([[0.], np.cumsum(seg)]); t /= t[-1]
#     deg = min(ft.POLY_DEG, len(p) - 1)
#     r = np.hypot(p[:, 0] - np.polyval(np.polyfit(t, p[:, 0], deg), t),
#                  p[:, 1] - np.polyval(np.polyfit(t, p[:, 1], deg), t))
#     print(f"       クリック点 {len(p)} 個 / 多項式次数 {deg}")
#     print(f"       点間距離 [m]: {np.round(seg, 3)}")
#     print(f"       各点の残差 [m]: {np.round(r, 4)}")
#     print(f"       残差 RMS = {np.sqrt(np.mean(r ** 2)):.4f} m")
#     if seg.min() < 1.0:
#         print(f"       警告: 最小間隔 {seg.min():.3f} m — 重複クリックの疑い")
#     if len(p) < deg + 3:
#         print(f"       警告: 点が少なく、残差では異常を検出できません")
#     print(f"       中心線の被覆 Y: {cxy[:,1].min():.2f} .. {cxy[:,1].max():.2f}")
#     print(f"       データの範囲 Y: {xy[:,1].min():.2f} .. {xy[:,1].max():.2f}")


# def assert_round_trip_real(xy, cxy, s, tan):
#     """C2 実データでの往復誤差。層A4と同じ主張を実データで確認する。"""
#     print("\n[C2] 往復誤差(実データ)")
#     s_out, d_out = ft.cart_to_frenet(xy, cxy, s, tan)
#     err = np.linalg.norm(frenet_to_cart(s_out, d_out, cxy, s, tan) - xy, axis=1)
#     tol = round_trip_tolerance(d_out, s, ROAD_RADIUS)
#     print(f"       mean {err.mean():.3e} / p99 {np.percentile(err,99):.3e} / "
#           f"max {err.max():.3e} m")
#     check("往復誤差", err, np.zeros(len(err)), tol,
#           f"(|d|max {np.abs(d_out).max():.1f} m, 間隔 {s[-1]/len(s)*100:.2f} cm)")


# def assert_perpendicular_foot(xy, cxy, s, tan):
#     """C3 最近傍サンプルが垂線の足になっているか.

#     判定量: rel·tg(接線方向の残差)。垂線の足なら定義上 0 なので、
#       サンプル間隔程度に収まるべき。内部の点と外側の点を分けて出すのが要点で、
#       全体の max だけ見ると外挿点に引きずられて実装の問題に見えてしまう。
#     """
#     print("\n[C3] 垂線の足(rel.tg)")
#     _, idx = cKDTree(cxy).query(xy, k=1)
#     t = np.abs(np.einsum("ij,ij->i", xy - cxy[idx], tan[idx]))
#     inside = (idx > 0) & (idx < len(cxy) - 1)
#     spacing = s[-1] / len(s)
#     print(f"       全点   : median {np.median(t):.4f} / max {t.max():.4f} m")
#     print(f"       内部のみ: max {t[inside].max():.4f} m "
#           f"({inside.sum()}/{len(xy)} 点, サンプル間隔 {spacing:.4f} m)")
#     check("内部の点の垂線残差", [t[inside].max()], [0.0], 3 * spacing)


# def report_reverse_steps(df, s_out):
#     """C4 進行方向を考慮した逆行率.

#     なぜ必要か: 対向車線の車は s が減る向きに走る。進行方向を無視して
#       diff < 0 を数えると、対向車線の全ステップが逆行に計上され、
#       20% 台の無意味な数字になる。車両ごとに符号を決めてから数えること。

#     得られるもの: 追跡ノイズの指標。数%なら正常。
#     """
#     print("\n[C4] 逆行率")
#     d = df.assign(_s=s_out)
#     naive = aware = tot = 0
#     for _, g in d.groupby("Vehicle_ID"):
#         o = g.sort_values("Frame")["_s"].to_numpy()
#         if len(o) < 2:
#             continue
#         st = np.diff(o)
#         sign = 1.0 if (o[-1] - o[0]) >= 0 else -1.0
#         naive += int((st < -0.05).sum())
#         aware += int((st * sign < -0.05).sum())
#         tot += len(st)
#     print(f"       進行方向を無視: {naive/tot:.3%}  <- 対向車線を含むため無意味")
#     print(f"       進行方向を考慮: {aware/tot:.3%}  <- 実際のノイズ指標")


# def assert_coverage(xy, cxy, s, tan, max_err, strict):
#     """C5 外挿量から、有効な s の上限を算出する.

#     層B2 で検証した ext^2/(2R) を使って、許容誤差 max_err から
#     逆算した s 上限を出す。テストが失敗を告げるだけでなく、
#     「ではどこまでなら使えるのか」を数値で返すのが狙い。
#     """
#     print("\n[C5] 被覆とクランプ")
#     _, idx = cKDTree(cxy).query(xy, k=1)
#     n_lo, n_hi = int((idx == 0).sum()), int((idx == len(cxy) - 1).sum())
#     frac = (n_lo + n_hi) / len(xy)
#     print(f"       中心線の外側: 手前端 {n_lo} / 奥端 {n_hi} = {frac:.2%}")
#     if n_hi == 0:
#         print("  OK   全点が中心線の内側")
#         return
#     s_out, _ = ft.cart_to_frenet(xy, cxy, s, tan)
#     far = float(s_out[idx == len(cxy) - 1].max())
#     ext = far - s[-1]
#     print(f"       クランプ開始 s = {s[-1]:.1f} m / 最遠 s = {far:.1f} m "
#           f"(外挿 {ext:.1f} m)")
#     print(f"       R={ROAD_RADIUS:.0f} m での最大外挿誤差 = {ext**2/(2*ROAD_RADIUS):.2f} m")
#     ext_ok = np.sqrt(2 * ROAD_RADIUS * max_err)
#     s_limit = s[-1] + ext_ok
#     n_bad = int((s_out > s_limit).sum())
#     print(f"  ==>  許容誤差 {max_err:.2f} m なら 有効な s 上限 = {s_limit:.1f} m")
#     print(f"       この上限を超える点: {n_bad} 点 ({n_bad/len(xy):.2%})")
#     if n_bad:
#         msg = (f"s > {s_limit:.1f} m の {n_bad} 点が許容誤差を超えます。"
#                f"中心線を延長するか解析範囲を制限してください")
#         print(f"  FAIL {msg}")
#         if strict:
#             _failures.append("coverage")


# def run_real(max_err, strict):
#     print("\n" + "=" * 62)
#     print(" 層C  実データ")
#     print("=" * 62)
#     raw = np.load(ft.CENTERLINE_NPY)
#     df = pd.read_csv(ft.CSV_PATH)
#     xy = df[["X_meter", "Y_meter"]].to_numpy(float)
#     cxy, s, tan = ft.fit_centerline(raw, n=ft.N_SAMPLES)
#     print(f"中心線 全長 {s[-1]:.2f} m / サンプル {len(cxy)} / "
#           f"データ {len(df)} 行 {df.Vehicle_ID.nunique()} 台")
#     report_centerline_quality(raw, cxy, s, xy)
#     assert_round_trip_real(xy, cxy, s, tan)
#     assert_perpendicular_foot(xy, cxy, s, tan)
#     report_reverse_steps(df, ft.cart_to_frenet(xy, cxy, s, tan)[0])
#     assert_coverage(xy, cxy, s, tan, max_err, strict)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--real", action="store_true", help="層Cも実行する")
#     ap.add_argument("--strict", action="store_true", help="層Cの違反で異常終了")
#     ap.add_argument("--max-extrap-error", type=float, default=0.20,
#                     help="許容する外挿誤差[m] (既定 0.20)")
#     a = ap.parse_args()

#     print("=" * 62); print(" 層A  実装の正しさ(合成データ)"); print("=" * 62)
#     for t in [test_straight_exact, test_circular_known_offsets,
#               test_off_sample_query, test_round_trip_synthetic,
#               test_sampling_stability, test_fit_centerline_straight,
#               test_fit_centerline_duplicate]:
#         t()
#     print("\n" + "=" * 62); print(" 層B  定義域の境界"); print("=" * 62)
#     test_beyond_ends_is_linear()
#     test_extrapolation_error_law()

#     if a.real:
#         run_real(a.max_extrap_error, a.strict)

#     print("\n" + "=" * 62)
#     if _failures:
#         print(f" 失敗 {len(_failures)} 件: {_failures}")
#         sys.exit(1)
#     print(" すべて合格")
#     print("=" * 62)


# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
"""test_frenet.py — Frenet変換(frenet.py)の検証スイート

============================================================================
 このテストが検証すること・しないこと
============================================================================
検証する : 中心線を「与えられたもの」としたときの変換の正しさ
検証しない: 中心線そのものが道路に沿っているか

  中心線を30度傾けても、その傾いた線で測って同じ線で戻せば往復は成立する。
  つまり本ファイルの検査は中心線の正しさから完全に独立している。
  中心線の妥当性は「車線維持車の d(s) が水平か」で別途確認すること
  (対向車線22本で -0.10 m/100m を確認済み)。

============================================================================
 構成
============================================================================
 層A 実装の正しさ   合成データ。解析解と比較する
   A1 test_straight_exact              直線: s=y, d=-x が厳密に出るか
   A2 test_circular_known_offsets      円弧: 既知の法線オフセットを復元するか
   A3 test_off_sample_query            格納点と一致しない点で検証(空テスト対策)
   A4 test_round_trip_synthetic        (s,d)->(X,Y) の往復
   A5 test_sampling_stability          サンプル数を変えても答えが変わらないか
   A6 test_fit_centerline_straight     fit_centerline が直線を直線に写すか
   A7 test_fit_centerline_duplicate    重複クリック点への耐性

 層B 定義域の境界   外挿の挙動と誤差の予測式
   B1 test_beyond_ends_is_linear       中心線の外は末端接線の直線になること
   B2 test_extrapolation_error_law     誤差が ext^2/(2R) に従うこと

 層C 実データ       層Bの式を使って許容範囲を判定
   C1 report_centerline_quality        クリック点の質(残差・重複・被覆)
   C2 assert_round_trip_real           往復誤差
   C3 assert_perpendicular_foot        垂線の足が取れているか
   C4 report_reverse_steps             進行方向を考慮した逆行率
   C5 assert_coverage                  外挿量から有効な s 上限を算出

============================================================================
 使い方
============================================================================
    python test_frenet.py                 # 層A+B のみ(高速・環境非依存)
    python test_frenet.py --real          # 層Cも実行(診断を表示)
    python test_frenet.py --real --strict # 層Cの違反で異常終了(CI用)
    python test_frenet.py --real --max-extrap-error 0.20
                                          # 許容する外挿誤差[m]を指定
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import frenet as ft

# 参考値のみ。実際の判定は estimate_radius() が中心線から推定した値を使う。
# ここは「推定値が桁違いにおかしくないか」の突き合わせにしか使わない。
ROAD_RADIUS = 1100.0   # 較正 1101 m / 画像計測 960 m
_failures = []


# ============================================================
# 補助
# ============================================================
def check(name, actual, expected, tol, note=""):
    err = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    ok = err <= tol
    tag = "OK  " if ok else "FAIL"
    print(f"  {tag} {name}: max err = {err:.4g} (tol {tol:.4g}) {note}")
    if not ok:
        _failures.append(name)
    return ok


def circular_centerline(radius=50.0, a0=0.0, a1=np.pi / 2, samples=20001):
    """解析解が既知の円弧中心線。s = R*theta, 法線オフセット = d。"""
    th = np.linspace(a0, a1, samples)
    cxy = radius * np.column_stack([np.cos(th), np.sin(th)])
    s = radius * (th - a0)
    tan = np.column_stack([-np.sin(th), np.cos(th)])
    return cxy, s, tan, th


def left_normal(tan):
    """接線に対する左手法線。d の正方向と一致する。"""
    return np.column_stack([-tan[:, 1], tan[:, 0]])


def frenet_to_cart(s_out, d_out, cxy, s, tan):
    """(s,d) -> (X,Y) の逆変換.

    順変換は KD木で足を選ぶが、逆変換では s しか手元に無いので
    searchsorted で足を引き直す。この非対称性が往復誤差の主因になる。
    """
    idx = np.clip(np.searchsorted(s, s_out), 0, len(s) - 1)
    tg = tan[idx]
    return cxy[idx] + (s_out - s[idx])[:, None] * tg + d_out[:, None] * left_normal(tg)


def estimate_radius(cxy, s, tail_frac=0.25, straight_tol=1e-6):
    """中心線そのものから曲率半径を推定する(符号付き, 単位 m).

    なぜ末端側だけを使うか:
      外挿誤差 ext^2/(2R) を支配するのは末端付近の曲率であって、
      中心線全体の平均曲率ではない。実データの中心線は多項式フィットの
      自由度により s=0 で R=-3791 m, s=88 で R=-896 m と 4 倍変動する。
      全体平均を使うと外挿誤差を過小評価する。

    原理:
      曲率の定義は kappa = d(theta)/ds (theta は接線の向き)。
      横方向ずれを多項式フィットする方法もあるが、末端で評価すると
      高次項が増幅されて急カーブで誤差が出る(R=50 m で 6% の過大評価)。
      接線角 theta を直接 s の 2 次でフィットし、その微分を末端で評価する:
          theta(s) = a*s^2 + b*s + c   ->   kappa(s) = 2*a*s + b
      円弧では theta は s の 1 次なので a=0 となり厳密。曲率が線形に
      変化する道路(クロソイド)では theta が 2 次なのでこれも厳密。
      外挿は末端から始まるので kappa(L) を返す。

    直線の扱い:
      「kappa がいくつ以下なら直線か」を恣意的に決めず、末端区間の全長 L に
      わたる横方向ずれ kappa*L^2/2 が straight_tol[m] 未満なら直線(inf)とする。
      判定基準を長さの次元で表すことで、スケールに依らない意味を持たせる。

    Returns
    -------
    float : 符号付き曲率半径。左カーブが正、右カーブが負、直線は +inf
    """
    n = len(cxy)
    i0 = max(0, int(n * (1.0 - tail_frac)))
    if n - i0 < 10:
        return np.inf
    seg = cxy[i0:]
    u = s[i0:] - s[i0]
    L = float(u[-1])
    if L <= 0:
        return np.inf
    dxy = np.diff(seg, axis=0)
    step = np.hypot(dxy[:, 0], dxy[:, 1])
    ok = step > 1e-12
    if ok.sum() < 8:
        return np.inf
    theta = np.unwrap(np.arctan2(dxy[ok, 1], dxy[ok, 0]))
    u_mid = 0.5 * (u[:-1] + u[1:])[ok]
    a, b, _ = np.polyfit(u_mid, theta, 2)
    kappa = 2.0 * a * L + b                    # kappa(u=L)
    if abs(kappa) * L * L / 2.0 < straight_tol:
        return np.inf
    return 1.0 / kappa


def round_trip_tolerance(d_out, s, radius):
    """往復誤差の許容値を誤差機構から導出する.

    逆変換は足を最大1サンプルずらす。曲線上ではそのずれが
    |d| に比例した誤差 (|d| * spacing / R) を生む。その10倍を上限とする。
    「たぶんこのくらい」で決めないこと。値が動いたとき、実装が変わったのか
    離散化が変わったのかを区別できなくなる。
    """
    spacing = s[-1] / len(s)
    return max(10.0 * float(np.abs(d_out).max()) * spacing / radius, 1e-9)


# ============================================================
# 層A  実装の正しさ
# ============================================================
def test_straight_exact():
    """A1 直線中心線 c(s)=(0,s) では s=y, d=-x が厳密に成立する.

    得られるもの: 符号の規約が「進行方向に向かって左が正」であることの確認。
      接線 (0,1) の左は -x 側なので d=-x。将来ここが反転したら
      LANE_CENTERS_D の並びが全部裏返るので、真っ先に落ちてほしい検査。
    """
    print("\n[A1] 直線中心線での厳密解")
    y = np.linspace(0.0, 100.0, 10001)
    cxy = np.column_stack([np.zeros_like(y), y])
    tan = np.tile([0.0, 1.0], (len(y), 1))
    pts = np.array([[0., 5.], [2., 10.], [-3., 25.], [1.5, 70.25]])
    a_s, a_d = ft.cart_to_frenet(pts, cxy, y.copy(), tan)
    check("s = y", a_s, pts[:, 1], 1e-10)
    check("d = -x (左が正)", a_d, -pts[:, 0], 1e-10)


def test_circular_known_offsets():
    """A2 円弧中心線で、既知の法線オフセットを復元できるか.

    得られるもの: 曲率がある場合の s(弧長)と d(符号付き横距離)の正しさ。
      実道路(R=1100 m)より遥かに急な R=50 m で試すことで、
      曲率に起因する誤差を意図的に増幅した状態で検証している。
    """
    print("\n[A2] 円弧中心線での既知オフセット復元")
    R = 50.0
    cxy, s, tan, _ = circular_centerline(radius=R)
    th = np.array([0.15, 0.45, 0.9, 1.25])
    base = R * np.column_stack([np.cos(th), np.sin(th)])
    tg = np.column_stack([-np.sin(th), np.cos(th)])
    off = np.array([-3.0, 0.0, 1.5, 4.0])
    pts = base + off[:, None] * left_normal(tg)
    a_s, a_d = ft.cart_to_frenet(pts, cxy, s, tan)
    spacing = s[-1] / len(s)
    check("s = R*theta", a_s, R * th, 3 * spacing, f"(間隔 {spacing*1000:.2f} mm)")
    check("d = 既知オフセット", a_d, off, 3 * spacing)


def test_off_sample_query():
    """A3 中心線サンプルと一致しない点で検証する.

    なぜ必要か: cart_to_frenet(cxy, cxy, ...) のようにサンプル点そのものを
      問い合わせると rel=0 が厳密に成立し、d=0 は計算せずとも決まる。
      「誤差が厳密に 0」というテストは通ったのではなく、何も試していない。
      ここではサンプル間の中点を使い、最近傍探索と接線補正を実際に働かせる。
    """
    print("\n[A3] サンプル点と一致しない点での検証(空テスト対策)")
    R = 50.0
    cxy, s, tan, _ = circular_centerline(radius=R, samples=5001)
    mid = 0.5 * (cxy[:-1] + cxy[1:])          # 弦の中点。円の内側に僅かに入る
    _, d_mid = ft.cart_to_frenet(mid, cxy, s, tan)
    sag = R * (1 - np.cos(np.pi / 2 / 5000 / 2))   # 弦の凹み量(理論値)
    print(f"       中点の |d| 最大 = {np.abs(d_mid).max():.3e} m "
          f"(弦の凹み理論値 {sag:.3e} m)")
    off = 1.234
    pts = mid + off * left_normal(tan[:-1])
    _, d_off = ft.cart_to_frenet(pts, cxy, s, tan)
    check("既知オフセット 1.234 m の復元", d_off, np.full(len(pts), off), 5e-3)


def test_round_trip_synthetic():
    """A4 (s,d) -> (X,Y) の往復.

    得られるもの: 最も強い単一の主張 ——
      「(s,d) は (X,Y) と同じ情報を持つ座標系の付け替えに過ぎない」。
      往復が成立するなら、変換の過程で情報は一切失われていない。
      ただし中心線が正しい場所にあるかは何も言っていないことに注意。
    """
    print("\n[A4] 逆変換の往復(合成)")
    R = 50.0
    cxy, s, tan, _ = circular_centerline(radius=R)
    th = np.linspace(0.05, 1.5, 200)
    base = R * np.column_stack([np.cos(th), np.sin(th)])
    tg = np.column_stack([-np.sin(th), np.cos(th)])
    off = np.linspace(-4, 4, 200)
    xy = base + off[:, None] * left_normal(tg)
    s_out, d_out = ft.cart_to_frenet(xy, cxy, s, tan)
    rec = frenet_to_cart(s_out, d_out, cxy, s, tan)
    err = np.linalg.norm(rec - xy, axis=1)
    tol = round_trip_tolerance(d_out, s, R)
    check("往復誤差", err, np.zeros(len(err)), tol,
          f"(|d|max {np.abs(d_out).max():.1f} m)")


def test_sampling_stability():
    """A5 中心線のサンプル数を変えても答えが変わらないか.

    注意: この検査は外挿の誤りに対しては盲目。中心線の外側の点では
      どのサンプル数でも同じ末端接線が使われるため、2つの誤った答えが
      一致して合格してしまう。境界の検査は層Bが担当する。
    """
    print("\n[A5] サンプリング安定性")
    R = 50.0
    th = np.array([0.2, 0.6, 1.0, 1.35])
    base = R * np.column_stack([np.cos(th), np.sin(th)])
    tg = np.column_stack([-np.sin(th), np.cos(th)])
    pts = base + np.array([2., -1., 3., -2.5])[:, None] * left_normal(tg)
    c = circular_centerline(radius=R, samples=3001)[:3]
    d = circular_centerline(radius=R, samples=30001)[:3]
    cr = ft.cart_to_frenet(pts, *c)
    dr = ft.cart_to_frenet(pts, *d)
    check("s の一致 (3001 vs 30001)", cr[0], dr[0], 2e-3)
    check("d の一致 (3001 vs 30001)", cr[1], dr[1], 2e-3)


def test_fit_centerline_straight():
    """A6 fit_centerline が直線クリック点から直線を作るか.

    得られるもの: 多項式フィット部分の健全性。既存テストは
      cart_to_frenet だけを試し、fit_centerline には assert が無かった。
    """
    print("\n[A6] fit_centerline: 直線の再現")
    pts = np.array([[0., 0.], [0., 25.], [0., 50.], [0., 75.], [0., 100.]])
    cxy, s, tan = ft.fit_centerline(pts, n=2001)
    check("横方向のふらつき", cxy[:, 0], np.zeros(len(cxy)), 1e-6)
    check("全長 = 100 m", [s[-1]], [100.0], 1e-6)
    check("接線 = (0,1)", tan, np.tile([0., 1.], (len(cxy), 1)), 1e-6)
    if np.any(np.diff(s) <= 0):
        _failures.append("s monotonic")
        print("  FAIL s が単調増加していません")
    else:
        print("  OK   s は単調増加")


def test_fit_centerline_duplicate():
    """A7 重複クリック点への耐性.

    なぜ必要か: 実データの centerline_meter.npy には 0.131 m しか
      離れていない2点がある(ダブルクリックの取りこぼし)。弦長パラメータ上で
      ほぼ同じ位置に2つの拘束が入り、末端が二重に重み付けされる。
    """
    print("\n[A7] fit_centerline: 重複クリック点への耐性")
    clean = np.array([[0., 0.], [0., 25.], [0., 50.], [0., 75.], [0., 100.]])
    dup = np.vstack([clean, clean[-1] + [0.05, 0.10]])
    c1, s1, _ = ft.fit_centerline(clean, n=2001)
    c2, s2, _ = ft.fit_centerline(dup, n=2001)
    dev = float(np.abs(c2[:, 0]).max())
    print(f"       横方向の歪み {dev:.4f} m / 全長差 {abs(s2[-1]-s1[-1]):.4f} m")
    check("重複による歪み", [dev], [0.0], 0.15)


# ============================================================
# 層B  定義域の境界
# ============================================================
def test_beyond_ends_is_linear():
    """B1 中心線の外側では末端接線の直線が基準線になることを明示する.

    なぜ必要か: cart_to_frenet は KD木で最近傍サンプルを引く。中心線より
      外の点にとって最近傍は必ず末端の1点になり、そこで idx が固定される。
      すると tg も末端接線に固定され、s_out = s[-1] + rel·tg は
      「末端接線に沿った直線」を基準にした値になる。
      既存テストは問い合わせ点を全て中心線の内側に置いていたため、
      この挙動を一度も踏んでいなかった。
    """
    print("\n[B1] 中心線の外側は直線外挿になる")
    R = 50.0
    cxy, s, tan, th = circular_centerline(radius=R)
    end_tan = tan[-1]
    for ext in [1.0, 3.0, 5.0]:
        pt = (cxy[-1] + ext * end_tan)[None, :]   # 末端接線上の点
        s_out, d_out = ft.cart_to_frenet(pt, cxy, s, tan)
        # 末端接線上なので、直線外挿なら d は厳密に 0、s は s[-1]+ext
        check(f"接線上 ext={ext:.0f} m の d", d_out, [0.0], 1e-6)
        check(f"接線上 ext={ext:.0f} m の s", s_out, [s[-1] + ext], 1e-6)


def test_extrapolation_error_law():
    """B2 外挿誤差が ext^2/(2R) に従うことを確認する.

    得られるもの: 実データで「どこまで外挿を許すか」を計算で決める根拠。
      層Cの assert_coverage はこの式を使って有効な s 上限を出すので、
      式そのものをここで検証しておかないと判断の根拠が閉じない。

      検査の作り方: 円弧を延長した先の点は「本来の道路上」にあるので
      真の d は 0。それを直線外挿した基準線で測った値が、そのまま外挿誤差。
    """
    print("\n[B2] 外挿誤差の法則 ext^2/(2R)")
    for R in [50.0, 300.0]:
        cxy, s, tan, th = circular_centerline(radius=R, a1=np.pi / 3)
        print(f"       R = {R:.0f} m")
        for ext in [2.0, 5.0, 10.0]:
            th_far = th[-1] + ext / R
            pt = R * np.array([[np.cos(th_far), np.sin(th_far)]])
            _, d_out = ft.cart_to_frenet(pt, cxy, s, tan)
            pred = ext ** 2 / (2 * R)
            got = abs(float(d_out[0]))
            rel = abs(got - pred) / pred
            tag = "OK  " if rel < 0.10 else "FAIL"
            if rel >= 0.10:
                _failures.append(f"extrap law R={R} ext={ext}")
            print(f"  {tag} ext={ext:5.1f} m: 実測 |d| = {got:.4f} m / "
                  f"予測 {pred:.4f} m (差 {rel:.1%})")


def test_estimate_radius_known():
    """B3 既知の曲率半径を estimate_radius が復元できるか.

    得られるもの: 層Cの ROAD_RADIUS 定数を廃して、中心線から自動推定する
      根拠。推定が正しいと確認できて初めて、実データの有効範囲の判定を
      推定値に委ねられる。

    検査対象:
      左カーブ(正), 右カーブ(負), 直線(inf), 実道路相当の緩いカーブ
    """
    print("\n[B3] estimate_radius: 既知の半径の復元")
    for R_true, label in [(50.0, "急な左カーブ"), (300.0, "中程度"),
                          (1100.0, "実道路相当")]:
        cxy, s, tan, _ = circular_centerline(radius=R_true,
                                             a1=min(np.pi / 2, 120.0 / R_true))
        got = estimate_radius(cxy, s)
        rel = abs(got - R_true) / R_true
        tag = "OK  " if rel < 0.02 else "FAIL"
        if rel >= 0.02:
            _failures.append(f"estimate_radius R={R_true}")
        print(f"  {tag} {label:12s} 真値 {R_true:7.1f} m -> 推定 {got:+8.1f} m "
              f"(差 {rel:.2%})")

    # 右カーブ(符号が反転すること)
    th = np.linspace(0, 120.0 / 300.0, 20001)
    cxy = 300.0 * np.column_stack([np.cos(th), -np.sin(th)])
    seg = np.hypot(*np.diff(cxy, axis=0).T)
    s = np.concatenate([[0.], np.cumsum(seg)])
    got = estimate_radius(cxy, s)
    tag = "OK  " if (got < 0 and abs(abs(got) - 300.0) / 300.0 < 0.02) else "FAIL"
    if tag == "FAIL":
        _failures.append("estimate_radius sign")
    print(f"  {tag} {'右カーブ':12s} 真値 -300.0 m -> 推定 {got:+8.1f} m "
          f"(符号が負であること)")

    # 直線
    y = np.linspace(0.0, 120.0, 20001)
    cxy = np.column_stack([np.zeros_like(y), y])
    got = estimate_radius(cxy, y.copy())
    tag = "OK  " if np.isinf(got) else "FAIL"
    if tag == "FAIL":
        _failures.append("estimate_radius straight")
    print(f"  {tag} {'直線':12s} 真値 inf     -> 推定 {got}")


def test_estimate_radius_varying_curvature():
    """B4 曲率が s に沿って変化する中心線で、末端の曲率を返すか.

    なぜ必要か: 実データの中心線は 3 次多項式フィットのため曲率が
      s=0 の R=-3791 m から s=88 の R=-896 m まで 4 倍変動している。
      外挿誤差を決めるのは末端の曲率なので、全体平均を返す実装だと
      誤差を過小評価して有効範囲を広く見積もってしまう。

      ここでは曲率が線形に増える曲線(クロソイド)を作り、
      推定値が「全体平均」ではなく「末端の値」に一致することを確認する。
    """
    print("\n[B4] estimate_radius: 曲率が変化する中心線")
    L = 120.0
    k0, k1 = 1.0 / 4000.0, 1.0 / 500.0        # 始点と終点の曲率
    u = np.linspace(0.0, L, 40001)
    kappa = k0 + (k1 - k0) * u / L
    theta = np.cumsum(np.concatenate([[0.0], np.diff(u) * kappa[:-1]]))
    cxy = np.column_stack([np.cumsum(np.concatenate([[0.0], np.diff(u) * np.cos(theta[:-1])])),
                           np.cumsum(np.concatenate([[0.0], np.diff(u) * np.sin(theta[:-1])]))])
    seg = np.hypot(*np.diff(cxy, axis=0).T)
    s = np.concatenate([[0.], np.cumsum(seg)])
    got = estimate_radius(cxy, s, tail_frac=0.25)
    R_end = 1.0 / k1                       # 末端(外挿が始まる点)の曲率半径
    R_mean = 1.0 / float(np.mean(kappa))   # 全体平均
    R_win0 = 1.0 / (k0 + (k1 - k0) * 0.75) # 末端区間の始点
    rel = abs(got - R_end) / R_end
    tag = "OK  " if rel < 0.05 else "FAIL"
    if rel >= 0.05:
        _failures.append("estimate_radius varying")
    print(f"       全体平均 R = {R_mean:7.1f} m / 末端区間の始点 R = {R_win0:7.1f} m")
    print(f"       末端(外挿の起点) R = {R_end:7.1f} m  <- これを返すべき")
    print(f"  {tag} 推定 {got:+8.1f} m (差 {rel:.1%})")
    for name, R_wrong in [("全体平均", R_mean), ("区間始点", R_win0)]:
        if abs(got - R_wrong) / R_wrong < 0.05:
            _failures.append(f"estimate_radius returned {name}")
            print(f"  FAIL {name}を返しています(末端を見ていない)")


# ============================================================
# 層C  実データ
# ============================================================
def report_centerline_quality(raw, cxy, s, xy):
    """C1 クリック点の質と被覆を報告する.

    得られるもの:
      - 残差: 特定の点だけ残差が大きければ、クリックミスか重複を示す
      - 最小間隔: 1 m を切っていればダブルクリックの疑い
      - 被覆: 中心線がデータ範囲を覆っているか。ここが外挿問題の根本原因
    """
    print("\n[C1] 中心線の質")
    p = np.asarray(raw, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    t = np.concatenate([[0.], np.cumsum(seg)]); t /= t[-1]
    deg = min(ft.POLY_DEG, len(p) - 1)
    r = np.hypot(p[:, 0] - np.polyval(np.polyfit(t, p[:, 0], deg), t),
                 p[:, 1] - np.polyval(np.polyfit(t, p[:, 1], deg), t))
    print(f"       クリック点 {len(p)} 個 / 多項式次数 {deg}")
    print(f"       点間距離 [m]: {np.round(seg, 3)}")
    print(f"       各点の残差 [m]: {np.round(r, 4)}")
    print(f"       残差 RMS = {np.sqrt(np.mean(r ** 2)):.4f} m")
    if seg.min() < 1.0:
        print(f"       警告: 最小間隔 {seg.min():.3f} m — 重複クリックの疑い")
    if len(p) < deg + 3:
        print(f"       警告: 点が少なく、残差では異常を検出できません")
    print(f"       中心線の被覆 Y: {cxy[:,1].min():.2f} .. {cxy[:,1].max():.2f}")
    print(f"       データの範囲 Y: {xy[:,1].min():.2f} .. {xy[:,1].max():.2f}")


def assert_round_trip_real(xy, cxy, s, tan, radius):
    """C2 実データでの往復誤差。層A4と同じ主張を実データで確認する。"""
    print("\n[C2] 往復誤差(実データ)")
    s_out, d_out = ft.cart_to_frenet(xy, cxy, s, tan)
    err = np.linalg.norm(frenet_to_cart(s_out, d_out, cxy, s, tan) - xy, axis=1)
    tol = round_trip_tolerance(d_out, s, radius)
    print(f"       mean {err.mean():.3e} / p99 {np.percentile(err,99):.3e} / "
          f"max {err.max():.3e} m")
    check("往復誤差", err, np.zeros(len(err)), tol,
          f"(|d|max {np.abs(d_out).max():.1f} m, 間隔 {s[-1]/len(s)*100:.2f} cm)")


def assert_perpendicular_foot(xy, cxy, s, tan):
    """C3 最近傍サンプルが垂線の足になっているか.

    判定量: rel·tg(接線方向の残差)。垂線の足なら定義上 0 なので、
      サンプル間隔程度に収まるべき。内部の点と外側の点を分けて出すのが要点で、
      全体の max だけ見ると外挿点に引きずられて実装の問題に見えてしまう。
    """
    print("\n[C3] 垂線の足(rel.tg)")
    _, idx = cKDTree(cxy).query(xy, k=1)
    t = np.abs(np.einsum("ij,ij->i", xy - cxy[idx], tan[idx]))
    inside = (idx > 0) & (idx < len(cxy) - 1)
    spacing = s[-1] / len(s)
    print(f"       全点   : median {np.median(t):.4f} / max {t.max():.4f} m")
    print(f"       内部のみ: max {t[inside].max():.4f} m "
          f"({inside.sum()}/{len(xy)} 点, サンプル間隔 {spacing:.4f} m)")
    check("内部の点の垂線残差", [t[inside].max()], [0.0], 3 * spacing)


def report_reverse_steps(df, s_out):
    """C4 進行方向を考慮した逆行率.

    なぜ必要か: 対向車線の車は s が減る向きに走る。進行方向を無視して
      diff < 0 を数えると、対向車線の全ステップが逆行に計上され、
      20% 台の無意味な数字になる。車両ごとに符号を決めてから数えること。

    得られるもの: 追跡ノイズの指標。数%なら正常。
    """
    print("\n[C4] 逆行率")
    d = df.assign(_s=s_out)
    naive = aware = tot = 0
    for _, g in d.groupby("Vehicle_ID"):
        o = g.sort_values("Frame")["_s"].to_numpy()
        if len(o) < 2:
            continue
        st = np.diff(o)
        sign = 1.0 if (o[-1] - o[0]) >= 0 else -1.0
        naive += int((st < -0.05).sum())
        aware += int((st * sign < -0.05).sum())
        tot += len(st)
    print(f"       進行方向を無視: {naive/tot:.3%}  <- 対向車線を含むため無意味")
    print(f"       進行方向を考慮: {aware/tot:.3%}  <- 実際のノイズ指標")


def report_radius(cxy, s):
    """C0 中心線から曲率半径を推定し、参考値と突き合わせる.

    ROAD_RADIUS は「この映像で測った値」であって、コードが依存すべき
    定数ではない。推定値を採用し、参考値との乖離が 2 倍を超えたときだけ
    警告する(中心線の取り違え等の事故検出)。
    """
    print("\n[C0] 曲率半径の推定")
    R = estimate_radius(cxy, s)
    print(f"       中心線末端から推定: {R:+.1f} m")
    print(f"       参考値 ROAD_RADIUS: {ROAD_RADIUS:.1f} m "
          f"(較正 1101 m / 画像計測 960 m)")
    if np.isinf(R):
        print("       -> 直線とみなします(外挿誤差ゼロ)")
        return R
    ratio = abs(R) / ROAD_RADIUS
    if ratio > 2.0 or ratio < 0.5:
        print(f"       警告: 参考値と {ratio:.1f} 倍乖離。中心線を確認してください")
    else:
        print(f"       参考値との比 {ratio:.2f} — 整合")
    return abs(R)


def assert_coverage(xy, cxy, s, tan, max_err, strict, radius):
    """C5 外挿量から、有効な s の上限を算出する.

    層B2 で検証した ext^2/(2R) を使って、許容誤差 max_err から
    逆算した s 上限を出す。テストが失敗を告げるだけでなく、
    「ではどこまでなら使えるのか」を数値で返すのが狙い。
    """
    print("\n[C5] 被覆とクランプ")
    _, idx = cKDTree(cxy).query(xy, k=1)
    n_lo, n_hi = int((idx == 0).sum()), int((idx == len(cxy) - 1).sum())
    frac = (n_lo + n_hi) / len(xy)
    print(f"       中心線の外側: 手前端 {n_lo} / 奥端 {n_hi} = {frac:.2%}")
    if n_hi == 0:
        print("  OK   全点が中心線の内側")
        return
    s_out, _ = ft.cart_to_frenet(xy, cxy, s, tan)
    far = float(s_out[idx == len(cxy) - 1].max())
    ext = far - s[-1]
    print(f"       クランプ開始 s = {s[-1]:.1f} m / 最遠 s = {far:.1f} m "
          f"(外挿 {ext:.1f} m)")
    if np.isinf(radius):
        print("       直線のため外挿誤差はゼロ。上限なし")
        return
    print(f"       R={radius:.0f} m での最大外挿誤差 = {ext**2/(2*radius):.2f} m")
    ext_ok = np.sqrt(2 * radius * max_err)
    s_limit = s[-1] + ext_ok
    n_bad = int((s_out > s_limit).sum())
    print(f"  ==>  許容誤差 {max_err:.2f} m なら 有効な s 上限 = {s_limit:.1f} m")
    print(f"       この上限を超える点: {n_bad} 点 ({n_bad/len(xy):.2%})")
    if n_bad:
        msg = (f"s > {s_limit:.1f} m の {n_bad} 点が許容誤差を超えます。"
               f"中心線を延長するか解析範囲を制限してください")
        print(f"  FAIL {msg}")
        if strict:
            _failures.append("coverage")


def run_real(max_err, strict):
    print("\n" + "=" * 62)
    print(" 層C  実データ")
    print("=" * 62)
    raw = np.load(ft.CENTERLINE_NPY)
    df = pd.read_csv(ft.CSV_PATH)
    xy = df[["X_meter", "Y_meter"]].to_numpy(float)
    cxy, s, tan = ft.fit_centerline(raw, n=ft.N_SAMPLES)
    print(f"中心線 全長 {s[-1]:.2f} m / サンプル {len(cxy)} / "
          f"データ {len(df)} 行 {df.Vehicle_ID.nunique()} 台")
    report_centerline_quality(raw, cxy, s, xy)
    radius = report_radius(cxy, s)
    assert_round_trip_real(xy, cxy, s, tan, radius)
    assert_perpendicular_foot(xy, cxy, s, tan)
    report_reverse_steps(df, ft.cart_to_frenet(xy, cxy, s, tan)[0])
    assert_coverage(xy, cxy, s, tan, max_err, strict, radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="層Cも実行する")
    ap.add_argument("--strict", action="store_true", help="層Cの違反で異常終了")
    ap.add_argument("--max-extrap-error", type=float, default=0.20,
                    help="許容する外挿誤差[m] (既定 0.20)")
    a = ap.parse_args()

    print("=" * 62); print(" 層A  実装の正しさ(合成データ)"); print("=" * 62)
    for t in [test_straight_exact, test_circular_known_offsets,
              test_off_sample_query, test_round_trip_synthetic,
              test_sampling_stability, test_fit_centerline_straight,
              test_fit_centerline_duplicate]:
        t()
    print("\n" + "=" * 62); print(" 層B  定義域の境界"); print("=" * 62)
    test_beyond_ends_is_linear()
    test_extrapolation_error_law()
    test_estimate_radius_known()
    test_estimate_radius_varying_curvature()

    if a.real:
        run_real(a.max_extrap_error, a.strict)

    print("\n" + "=" * 62)
    if _failures:
        print(f" 失敗 {len(_failures)} 件: {_failures}")
        sys.exit(1)
    print(" すべて合格")
    print("=" * 62)


if __name__ == "__main__":
    main()