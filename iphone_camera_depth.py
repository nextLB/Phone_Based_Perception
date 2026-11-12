from camera_sensor_streamer import CameraStreamer
from shared_tools import update_fps
import time
import cv2


CAMERA_ID = 0
CAMERA_SIZE = "1280x960"    # 最好是 4：3
CAMERA_MAX_FPS = 30
CAMERA_BITRATE = 20000000  # 20 Mbps
CAMERA_CALIBRATION_SZIE_WIDTH = 1280
CAMERA_CALIBRATION_SZIE_HEIGHT = 960

RESIZE_WIDTH = 1280 
RESIZE_HEIGHT = 960





def main():

    # Camera
    cam = CameraStreamer(camera_id=CAMERA_ID, camera_size=CAMERA_SIZE, max_fps=CAMERA_MAX_FPS, bitrate=CAMERA_BITRATE)
    cam.start()
    cal = CameraStreamer.get_camera_calibration_from_file("oppo_reno14_pro_media_camera.txt", camera_id=CAMERA_ID)
    if cal is not None:
        K = cal["K"]
        print("内参矩阵 K:\n", K)
        print("外参旋转:", cal["pose_rotation"])
        print("外参平移:", cal["pose_translation"])
        print("active array:", cal["active_array"])

        # 如果 active array 与你当前流分辨率不同，先缩放再用
        if cal["active_array"] is not None:
            K_stream = CameraStreamer.scale_intrinsics(
                K, cal["active_array"], (CAMERA_CALIBRATION_SZIE_WIDTH, CAMERA_CALIBRATION_SZIE_HEIGHT)
            )
            print(f"缩放到 {CAMERA_CALIBRATION_SZIE_WIDTH}x{CAMERA_CALIBRATION_SZIE_HEIGHT} 的 K:\n", K_stream)
    else:
        print("文件里未解析到内参，需 OpenCV 标定")





    fps_frame_count = 0
    fps_start_time = time.time()
    current_fps = None

    print("主循环开始，按 'q' 退出...")
    print("-" * 60)
    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Stream", RESIZE_WIDTH, RESIZE_HEIGHT)



    try:
        while True:
            # 获取最新帧
            rgb_image, host_ts_ns = cam.get_latest_frame(timeout=0.01)


            # 绘制与显示
            if rgb_image is not None:
                # FPS 统计（满 1 秒才返回 current_fps，否则为 None）
                fps_frame_count, fps_start_time, current_fps = update_fps(fps_frame_count, fps_start_time)

                # --- 终端打印逻辑（每秒打印一次） ---
                if current_fps is not None:
                    print(f"[FPS] 实时帧率: {current_fps:.1f} fps")

                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                cv2.imshow("Camera Stream", bgr_image)


            # 处理退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break



    except KeyboardInterrupt:
        print("\n用户中断")
        
    finally:
        print("正在停止所有线程...")
        cam.stop()
        cv2.destroyAllWindows()
        print("清理完毕，程序退出。")








if __name__ == "__main__":
    main()



