# capture_frames.py
import os
import time
import json
import cv2
import numpy as np
from camera_sensor_streamer import CameraStreamer

# ============================== 配置 ==============================
CAMERA_ID = 0
CAMERA_SIZE = "1280x960"           # 必须 4:3
CAMERA_MAX_FPS = 30
CAMERA_BITRATE = 20000000
CALIB_FILE = "oppo_reno14_pro_media_camera.txt"

WARMUP_SECONDS = 5.0               # 预热时间（让相机自动曝光/对焦稳定）
CAPTURE_COUNT = 100                # 采集张数（不用太多，几十张就够做两视图/多视图）
CAPTURE_INTERVAL = 0.2             # 采集间隔（秒），期间请缓慢平移手机
CAPTURE_TIMEOUT = 300.0            # 超时保护

# ★ 输出到一个新文件夹，避免和旧数据混淆
OUTPUT_DIR = "D:/LB/Datasets/iphone_capture"      
FRAME_EXT = "png"

# ============================== 工具 ==============================
def parse_size(size_str):
    w, h = size_str.lower().split("x")
    return int(w), int(h)

def put_lines(img, lines, color=(0, 255, 0)):
    y = 32
    for line in lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        y += 32
    return img

# ============================== 主流程 ==============================
def main():
    cap_w, cap_h = parse_size(CAMERA_SIZE)
    print(f"[INFO] 采集分辨率: {cap_w}x{cap_h}，目标张数: {CAPTURE_COUNT}，间隔: {CAPTURE_INTERVAL}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    # -------- 1. 启动相机 --------
    cam = CameraStreamer(camera_id=CAMERA_ID, camera_size=CAMERA_SIZE, max_fps=CAMERA_MAX_FPS, bitrate=CAMERA_BITRATE)
    cam.start()

    # -------- 2. 解析内参（源头已自动纠偏） --------
    cal = CameraStreamer.get_camera_calibration_from_file(CALIB_FILE, camera_id=CAMERA_ID)
    K_stream = None
    if cal is not None:
        if cal["active_array"] is not None:
            K_stream = CameraStreamer.scale_intrinsics(cal["K"], cal["active_array"], (cap_w, cap_h))
        else:
            K_stream = cal["K"]
        print(f"[CAL] 最终缩放后 ({cap_w}x{cap_h}) K:\n{K_stream}")
    else:
        print("[CAL] 警告: 未解析到内参，采集会继续但内参文件为空")

    win_name = f"Capture | {cap_w}x{cap_h} | 0/{CAPTURE_COUNT}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cap_w // 2, cap_h // 2)

    saved = 0
    captured_info = []

    try:
        # ==================== 3. 预热 ====================
        print("[INFO] 等待相机首帧...")
        first_frame_time = None
        while first_frame_time is None:
            frame, _ = cam.get_latest_frame(timeout=0.1)
            if frame is not None:
                first_frame_time = time.time()
                print(f"[INFO] 相机已出图，开始预热 {WARMUP_SECONDS} 秒...")
            if (cv2.waitKey(1) & 0xFF) == ord('q'): return

        while True:
            left = WARMUP_SECONDS - (time.time() - first_frame_time)
            if left <= 0: break
            frame, _ = cam.get_latest_frame(timeout=0.05)
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                put_lines(bgr, [f"Resolution: {cap_w}x{cap_h}", f"Warmup: {left:.1f}s"], color=(0, 200, 255))
                cv2.imshow(win_name, bgr)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): return

        # ==================== 4. 采集 ====================
        print(f"[INFO] 开始采集！请【平移】手机产生视差（不要只原地转头）")
        last_capture = 0.0
        start_capture = time.time()

        while saved < CAPTURE_COUNT:
            if time.time() - start_capture > CAPTURE_TIMEOUT: break

            frame, ts = cam.get_latest_frame(timeout=0.05)
            if frame is not None:
                now = time.time()
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if now - last_capture >= CAPTURE_INTERVAL:
                    last_capture = now
                    fname = f"frame_{saved:04d}.{FRAME_EXT}"
                    fpath = os.path.join(img_dir, fname)
                    cv2.imwrite(fpath, bgr)
                    saved += 1
                    captured_info.append({"file": fname, "timestamp_ns": int(ts) if ts is not None else None})
                    print(f"[{saved}/{CAPTURE_COUNT}] saved {fname}")
                    cv2.setWindowTitle(win_name, f"Capture | {saved}/{CAPTURE_COUNT}")

                put_lines(bgr, [f"Captured: {saved}/{CAPTURE_COUNT}", f"Interval: {CAPTURE_INTERVAL}s"])
                cv2.imshow(win_name, bgr)

            if (cv2.waitKey(1) & 0xFF) == ord('q'): break

        # ==================== 5. 保存内参文件 ====================
        dist_default = [0.0, 0.0, 0.0, 0.0, 0.0]
        intr = {
            "camera_id": CAMERA_ID,
            "camera_size_str": CAMERA_SIZE,
            "image_width": cap_w,
            "image_height": cap_h,
            "K": K_stream.tolist() if K_stream is not None else None,
            "dist_coeffs": dist_default,
            "dist_note": "手机 static info 无畸变参数，此处为 0；需要精确畸变请用棋盘格 OpenCV 标定后替换",
            "active_array": cal["active_array"] if cal else None,
            "raw_intrinsic": cal["raw_intrinsic"] if cal else None,
            "raw_K": cal["K"].tolist() if cal else None,
            "pose_rotation": cal["pose_rotation"] if cal else None,
            "pose_translation": cal["pose_translation"] if cal else None,
            "captured_count": saved,
            "captures": captured_info,
        }
        intr_path = os.path.join(OUTPUT_DIR, "intrinsics.json")
        with open(intr_path, "w", encoding="utf-8") as f:
            json.dump(intr, f, indent=2, ensure_ascii=False)

        if K_stream is not None:
            yml_path = os.path.join(OUTPUT_DIR, "intrinsics.yml")
            fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_WRITE)
            fs.write("image_width", cap_w)
            fs.write("image_height", cap_h)
            fs.write("K", K_stream)
            fs.write("dist", np.array(dist_default))
            fs.release()

        print(f"[DONE] 共采集 {saved} 张 → {img_dir}")
        print(f"[DONE] 内参文件 → {intr_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C 中断")
    finally:
        cam.stop()
        time.sleep(0.1)
        cv2.destroyAllWindows()
        print("[DONE] 程序退出")

if __name__ == "__main__":
    main()