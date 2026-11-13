# capture_frames.py
import os
import time
import json
import cv2
import numpy as np
from camera_sensor_streamer import CameraStreamer
from sensor_streamer import SensorStreamer

# ============================== 配置 ==============================
CAMERA_ID = 0
CAMERA_SIZE = "1280x960"           # 必须 4:3
CAMERA_MAX_FPS = 30
CAMERA_BITRATE = 20000000
CALIB_FILE = "oppo_reno14_pro_media_camera.txt"

PHONE_IP = "192.168.1.145"         # 与 main.py 一致

WARMUP_SECONDS = 5.0
CAPTURE_COUNT = 500
CAPTURE_INTERVAL = 0.1
CAPTURE_TIMEOUT = 300.0

OUTPUT_DIR = "D:/LB/Datasets/iphone_capture"
FRAME_EXT = "png"
FRAME_META_FILE = "frames.jsonl"   # ★ 新增：每帧图像 + IMU 快照

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

def fmt_vec(v):
    if v is None:
        return "None"
    return f"[{v[0]:7.3f}, {v[1]:7.3f}, {v[2]:7.3f}]"

# ============================== 主流程 ==============================
def main():
    cap_w, cap_h = parse_size(CAMERA_SIZE)
    print(f"[INFO] 采集分辨率: {cap_w}x{cap_h}，目标张数: {CAPTURE_COUNT}，间隔: {CAPTURE_INTERVAL}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    frames_log_path = os.path.join(OUTPUT_DIR, FRAME_META_FILE)
    # 清空旧的 frames.jsonl，避免多轮采集混在一起
    with open(frames_log_path, "w", encoding="utf-8") as f:
        pass

    # -------- 1. 启动相机 --------
    cam = CameraStreamer(camera_id=CAMERA_ID, camera_size=CAMERA_SIZE, max_fps=CAMERA_MAX_FPS, bitrate=CAMERA_BITRATE)
    cam.start()

    # -------- 1.5 启动传感器（新增） --------
    print("[SENSOR] 启动 IMU 传感器...")
    acc_sensor  = SensorStreamer("android.sensor.accelerometer",   phone_ip=PHONE_IP)
    gyro_sensor = SensorStreamer("android.sensor.gyroscope",       phone_ip=PHONE_IP)
    mag_sensor  = SensorStreamer("android.sensor.magnetic_field",  phone_ip=PHONE_IP)
    acc_sensor.start()
    gyro_sensor.start()
    mag_sensor.start()

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

    # ★ 传感器最新值缓存（每轮循环都尝试刷新，保存帧时直接取用）
    latest_acc = None
    latest_gyro = None
    latest_mag = None

    def refresh_sensors():
        """每轮循环调用，保持 latest_* 是最新值。"""
        nonlocal latest_acc, latest_gyro, latest_mag
        a, _, _ = acc_sensor.get_latest_data(timeout=0.0)
        g, _, _ = gyro_sensor.get_latest_data(timeout=0.0)
        m, _, _ = mag_sensor.get_latest_data(timeout=0.0)
        if a is not None: latest_acc = a
        if g is not None: latest_gyro = g
        if m is not None: latest_mag = m

    try:
        # ==================== 3. 预热 ====================
        print("[INFO] 等待相机首帧...")
        first_frame_time = None
        while first_frame_time is None:
            refresh_sensors()
            frame, _ = cam.get_latest_frame(timeout=0.1)
            if frame is not None:
                first_frame_time = time.time()
                print(f"[INFO] 相机已出图，开始预热 {WARMUP_SECONDS} 秒...")
            if (cv2.waitKey(1) & 0xFF) == ord('q'): return

        while True:
            refresh_sensors()
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

            refresh_sensors()                       # 先刷新传感器
            frame, ts = cam.get_latest_frame(timeout=0.05)

            if frame is not None:
                now = time.time()
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if now - last_capture >= CAPTURE_INTERVAL:
                    last_capture = now
                    fname = f"frame_{saved:04d}.{FRAME_EXT}"
                    fpath = os.path.join(img_dir, fname)
                    cv2.imwrite(fpath, bgr)

                    # ★ 组装一帧的完整记录
                    record = {
                        "frame": fname,
                        "frame_index": saved,
                        "cam_ts_ns":  int(ts) if ts is not None else None,   # 相机帧时间戳
                        "host_ts_ns": int(now * 1e9),                        # 保存时刻 host 时间
                        "acc":  [float(x) for x in latest_acc]  if latest_acc  is not None else None,
                        "gyro": [float(x) for x in latest_gyro] if latest_gyro is not None else None,
                        "mag":  [float(x) for x in latest_mag]  if latest_mag  is not None else None,
                    }
                    with open(frames_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    saved += 1
                    captured_info.append({
                        "file": fname,
                        "frame_index": saved - 1,
                        "timestamp_ns": int(ts) if ts is not None else None,
                    })
                    print(f"[{saved}/{CAPTURE_COUNT}] saved {fname}  "
                          f"ACC={fmt_vec(latest_acc)} GYRO={fmt_vec(latest_gyro)} MAG={fmt_vec(latest_mag)}")
                    cv2.setWindowTitle(win_name, f"Capture | {saved}/{CAPTURE_COUNT}")

                put_lines(bgr, [
                    f"Captured: {saved}/{CAPTURE_COUNT}",
                    f"Interval: {CAPTURE_INTERVAL}s",
                    f"ACC : {fmt_vec(latest_acc)}",
                    f"GYRO: {fmt_vec(latest_gyro)}",
                    f"MAG : {fmt_vec(latest_mag)}",
                ])
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
            "frame_meta_file": FRAME_META_FILE,      # ★ 指向 frames.jsonl
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
        print(f"[DONE] 帧+IMU 元数据 → {frames_log_path}")
        print(f"[DONE] 内参文件 → {intr_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C 中断")
    finally:
        cam.stop()
        acc_sensor.stop()
        gyro_sensor.stop()
        mag_sensor.stop()
        time.sleep(0.1)
        cv2.destroyAllWindows()
        print("[DONE] 程序退出")

if __name__ == "__main__":
    main()