from camera_sensor_streamer import CameraStreamer
from shared_tools import update_fps
from sensor_streamer import SensorStreamer
from location_streamer import LocationStreamer
import time
import cv2


CAMERA_ID = 4
CAMERA_SIZE = "1280x960"    # 最好是 4：3
CAMERA_MAX_FPS = 30
CAMERA_BITRATE = 20000000  # 20 Mbps
CAMERA_CALIBRATION_SZIE_WIDTH = 1280
CAMERA_CALIBRATION_SZIE_HEIGHT = 960

RESIZE_WIDTH = 1280 
RESIZE_HEIGHT = 960

PHONE_IP = "192.168.1.145"





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


    # SensorS
    acc_sensor = SensorStreamer("android.sensor.accelerometer", phone_ip=PHONE_IP)
    gyro_sensor = SensorStreamer("android.sensor.gyroscope", phone_ip=PHONE_IP)
    mag_sensor = SensorStreamer("android.sensor.magnetic_field", phone_ip=PHONE_IP)
    acc_sensor.start()
    gyro_sensor.start()
    mag_sensor.start()


    # GPS
    gps_sensor = LocationStreamer(poll_interval=0.1)
    gps_sensor.start()


    fps_frame_count = 0
    fps_start_time = time.time()
    current_fps = None

    latest_data = {"acc": None, "gyro": None, "mag": None, "gps": None}

    print("主循环开始，按 'q' 退出...")
    print("-" * 60)
    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Stream", RESIZE_WIDTH, RESIZE_HEIGHT)


    try:
        while True:
            # 获取最新帧
            rgb_image, host_ts_ns = cam.get_latest_frame(timeout=0.01)

            # 获取传感器数据
            acc_vals, _, _ = acc_sensor.get_latest_data(timeout=0.0)
            if acc_vals is not None: latest_data["acc"] = acc_vals
                
            gyro_vals, _, _ = gyro_sensor.get_latest_data(timeout=0.0)
            if gyro_vals is not None: latest_data["gyro"] = gyro_vals
                
            mag_vals, _, _ = mag_sensor.get_latest_data(timeout=0.0)
            if mag_vals is not None: latest_data["mag"] = mag_vals

            gps_src, gps_lat, gps_lon, gps_ts = gps_sensor.get_latest_location(timeout=0.0)
            if gps_lat is not None:
                latest_data["gps"] = (gps_src, gps_lat, gps_lon)


            # 绘制与显示
            if rgb_image is not None:
                # FPS 统计（满 1 秒才返回 current_fps，否则为 None）
                fps_frame_count, fps_start_time, current_fps = update_fps(fps_frame_count, fps_start_time)

                # --- 终端打印逻辑（每秒打印一次） ---
                if current_fps is not None:
                    print(f"[FPS] 实时帧率: {current_fps:.1f} fps")

                    if latest_data["acc"]:
                        print(f"  [ACC ] x={latest_data['acc'][0]:7.3f}, y={latest_data['acc'][1]:7.3f}, z={latest_data['acc'][2]:7.3f}")
                    if latest_data["gyro"]:
                        print(f"  [GYRO] x={latest_data['gyro'][0]:7.3f}, y={latest_data['gyro'][1]:7.3f}, z={latest_data['gyro'][2]:7.3f}")
                    if latest_data["mag"]:
                        print(f"  [MAG ] x={latest_data['mag'][0]:7.3f}, y={latest_data['mag'][1]:7.3f}, z={latest_data['mag'][2]:7.3f}")

                    if latest_data["gps"]:
                        src, lat, lon = latest_data["gps"]
                        print(f"  [GPS ] [{src}] lat={lat}, lon={lon}")
                    else:
                        print(f"  [GPS ] 等待定位数据... (请在手机打开地图App)")



                    print("-" * 60)


                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


                # --- 画面叠加数据 ---
                text_y = 30 
                if latest_data["acc"]:
                    txt = f"ACC: {latest_data['acc'][0]:6.2f}, {latest_data['acc'][1]:6.2f}, {latest_data['acc'][2]:6.2f}"
                    cv2.putText(bgr_image, txt, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    text_y += 30
                    
                if latest_data["gyro"]:
                    txt = f"GYRO: {latest_data['gyro'][0]:6.2f}, {latest_data['gyro'][1]:6.2f}, {latest_data['gyro'][2]:6.2f}"
                    cv2.putText(bgr_image, txt, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    text_y += 30
                    
                if latest_data["mag"]:
                    txt = f"MAG: {latest_data['mag'][0]:6.2f}, {latest_data['mag'][1]:6.2f}, {latest_data['mag'][2]:6.2f}"
                    cv2.putText(bgr_image, txt, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    text_y += 30

                if latest_data["gps"]:
                    src, lat, lon = latest_data["gps"]
                    txt = f"GPS({src}): {lat}, {lon}"
                    cv2.putText(bgr_image, txt, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                



                cv2.imshow("Camera Stream", bgr_image)


                # TODO: 后续可能会做深度图推理估计等






            # 处理退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break



    except KeyboardInterrupt:
        print("\n用户中断")
        
    finally:
        print("正在停止所有线程...")
        cam.stop()
        acc_sensor.stop()
        gyro_sensor.stop()
        mag_sensor.stop()
        gps_sensor.stop()
        
        cam.join(timeout=2.0)
        acc_sensor.join(timeout=1.0)
        gyro_sensor.join(timeout=1.0)
        mag_sensor.join(timeout=1.0)
        gps_sensor.join(timeout=1.0)

        cv2.destroyAllWindows()
        print("清理完毕，程序退出。")






if __name__ == "__main__":
    main()



