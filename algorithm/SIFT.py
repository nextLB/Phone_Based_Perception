# SIFT.py
# 两视图 SfM：SIFT -> 本质矩阵 -> recoverPose -> 三角化 -> 点云 + 重投影可视化
import os
import json
import cv2
import numpy as np
import open3d as o3d  # ★ 引入 Open3D

# ============================== 配置 ==============================
DATA_DIR  = r"D:\LB\Datasets\iphone_capture"
IMG_DIR   = os.path.join(DATA_DIR, "images")
INTR_PATH = os.path.join(DATA_DIR, "intrinsics.json")
OUT_DIR   = os.path.join(DATA_DIR, "sfm_out")

FRAME1 = "frame_0000.png"
FRAME2 = "frame_0015.png"

RATIO         = 0.75
RANSAC_THRESH = 1.0
REPROJ_THRESH = 4.0
MIN_DEPTH     = 0.05

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

os.makedirs(OUT_DIR, exist_ok=True)

# ============================== 1. 内参 ==============================
with open(INTR_PATH, "r", encoding="utf-8") as f:
    intr = json.load(f)

K    = np.array(intr["K"], dtype=np.float64)
dist = np.array(intr["dist_coeffs"], dtype=np.float64)
W_img = intr["image_width"]
H_img = intr["image_height"]

print("=" * 64)
print("内参 (源头已自动纠偏)")
print("=" * 64)
print(f"分辨率: {W_img} x {H_img}")
print(f"K =\n{K}")

# ============================== 2. 读图 ==============================
p1 = os.path.join(IMG_DIR, FRAME1)
p2 = os.path.join(IMG_DIR, FRAME2)

g1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
g2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
assert g1 is not None, f"读图失败: {p1}"
assert g2 is not None, f"读图失败: {p2}"

vis_bgr1 = cv2.imread(p1)
vis_bgr2 = cv2.imread(p2)

print(f"\n图像: {FRAME1} <-> {FRAME2}")

# ============================== 3. SIFT + ratio test ==============================
sift = cv2.SIFT_create()
kp1, d1 = sift.detectAndCompute(g1, None)
kp2, d2 = sift.detectAndCompute(g2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
knn = bf.knnMatch(d1, d2, k=2)
good = [m[0] for m in knn if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
print(f"SIFT 关键点: kp1={len(kp1)}, kp2={len(kp2)}, good matches = {len(good)}")

if len(good) < 40:
    raise RuntimeError("匹配太少，换 pair 或放宽 RATIO")

match_img = cv2.drawMatches(vis_bgr1, kp1, vis_bgr2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
match_path = os.path.join(OUT_DIR, "matches.png")
cv2.imwrite(match_path, match_img)

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# ============================== 4. 本质矩阵 + recoverPose ==============================
print("\n开始多 seed 尝试...")
best = None
for seed in SEEDS:
    cv2.setRNGSeed(seed)
    E, maskE = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, RANSAC_THRESH)
    if E is None or maskE is None: continue
    nE = int(maskE.sum())
    nPose, R, t, maskPose = cv2.recoverPose(E, pts1, pts2, K, mask=maskE.copy())
    print(f"  seed={seed}: E_in={nE}, pose_in={nPose}")
    if best is None or nPose > best[0]:
        best = (nPose, E, R, t, maskE, maskPose, seed, nE)

if best is None: raise RuntimeError("所有 seed 均失败")

nPose, E, R, t, maskE, maskPose, best_seed, nE = best
print(f"\n[BEST] seed={best_seed}, E_in={nE}, pose_in={nPose}")
print(f"R =\n{R}")
print(f"t = {t.ravel()}")

if nPose < 50: raise RuntimeError(f"最佳 recoverPose 内点仅 {nPose}，视差不足")

# ============================== 5. 三角化 ==============================
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = K @ np.hstack([R, t])
pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
pts3D = (pts4D[:3] / pts4D[3]).T

# ============================== 6. 过滤 ==============================
cloud, colors, kept_idx = [], [], []
for i in range(len(pts3D)):
    if maskPose[i, 0] == 0: continue
    X = pts3D[i]
    if X[2] < MIN_DEPTH: continue
    proj = P2 @ np.array([X[0], X[1], X[2], 1.0])
    proj2d = proj[:2] / proj[2]
    if np.linalg.norm(proj2d - pts2[i]) > REPROJ_THRESH: continue

    cloud.append(X)
    kept_idx.append(i)
    u = int(np.clip(pts1[i][0], 0, vis_bgr1.shape[1] - 1))
    v = int(np.clip(pts1[i][1], 0, vis_bgr1.shape[0] - 1))
    b, g, r = vis_bgr1[v, u]
    colors.append((int(r), int(g), int(b)))

cloud    = np.array(cloud,    dtype=np.float64).reshape(-1, 3)
colors   = np.array(colors,   dtype=np.uint8).reshape(-1, 3)
kept_idx = np.array(kept_idx, dtype=np.int64)
print(f"\n最终点云点数: {len(cloud)}")
if len(cloud) == 0: raise RuntimeError("点云为空")

# ============================== 7. 重投影可视化 ==============================
X_h = np.hstack([cloud, np.ones((len(cloud), 1))]).T
proj1 = P1 @ X_h
proj1_2d = (proj1[:2] / proj1[2]).T
proj2 = P2 @ X_h
proj2_2d = (proj2[:2] / proj2[2]).T

vis1 = vis_bgr1.copy()
for i in kept_idx: cv2.circle(vis1, (int(pts1[i][0]), int(pts1[i][1])), 4, (0, 255, 0), 1, cv2.LINE_AA)
for p in proj1_2d: cv2.circle(vis1, (int(round(p[0])), int(round(p[1]))), 4, (0, 0, 255), 1, cv2.LINE_AA)

vis2 = vis_bgr2.copy()
for i in kept_idx: cv2.circle(vis2, (int(pts2[i][0]), int(pts2[i][1])), 4, (0, 255, 0), 1, cv2.LINE_AA)
for p in proj2_2d: cv2.circle(vis2, (int(round(p[0])), int(round(p[1]))), 4, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imwrite(os.path.join(OUT_DIR, "reproj_in_img1.png"), vis1)
cv2.imwrite(os.path.join(OUT_DIR, "reproj_in_img2.png"), vis2)

err1 = np.linalg.norm(proj1_2d - pts1[kept_idx], axis=1)
err2 = np.linalg.norm(proj2_2d - pts2[kept_idx], axis=1)
print(f"重投影误差 img1: mean={err1.mean():.3f}px, max={err1.max():.3f}px")
print(f"重投影误差 img2: mean={err2.mean():.3f}px, max={err2.max():.3f}px")

# ============================== 8. 保存点云 ==============================
ply_path = os.path.join(OUT_DIR, "cloud.ply")
with open(ply_path, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(cloud)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")
    for (x, y, z), (r, g, b) in zip(cloud, colors):
        f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

np.savez(os.path.join(OUT_DIR, "pose.npz"), R=R, t=t, K=K, E=E)
print(f"\n[DONE] 点云已保存至: {ply_path}")

# ============================== 9. Open3D 3D 可视化 ==============================
print("\n[INFO] 正在启动 Open3D 可视化...")
print("       鼠标左键：旋转视角")
print("       鼠标滚轮：缩放")
print("       鼠标中键：平移视角")
print("       按 'Q' 键：退出窗口")

# 读取点云
pcd = o3d.io.read_point_cloud(ply_path)

# 创建一个 3D 坐标系（原点），方便看清世界中心
# 长度设为 0.2，红轴=X，绿轴=Y，蓝轴=Z
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

# 把点云和坐标系一起显示
o3d.visualization.draw_geometries([pcd, coord_frame])