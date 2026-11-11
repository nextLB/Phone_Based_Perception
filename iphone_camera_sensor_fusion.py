import queue
import threading
import time
import cv2
import websocket
import json
import subprocess
import re

from adbutils import adb
from mysc.core.video import VideoAdapter, VideoKwargs, CameraKwargs

class CameraStreamer(threading.Thread):
    def __init__(self, camera_id=0, camera_size="1920x1080", max_fps=30, bitrate=20000000):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.camera_size = camera_size
        self.max_fps = max_fps
        self.bitrate = bitrate

        # 队列长度设为 1：主循环没取走旧帧时，新帧会自动顶替它，绝不积压延迟
        self.frame_queue = queue.Queue(maxsize=1)
        
        self.running = threading.Event()
        self.running.set()
        self.va = None

    def run(self):
        """后台线程：只负责死循环拿图，打时间戳，塞进队列，不参与任何绘制。"""
        print(f"[CAM] 正在连接相机 id={self.camera_id}...")
        try:
            device = adb.device_list()[0]
            camera_kwargs = CameraKwargs(camera_id=self.camera_id, camera_size=self.camera_size)
            video_kwargs = VideoKwargs(
                video_codec=VideoKwargs.EnumVideoCodec.H264,
                video_source=VideoKwargs.EnumVideoSource.CAMERA,
                max_fps=self.max_fps,
                video_bit_rate=self.bitrate,
                _camera_kwargs=camera_kwargs,
            )
            self.va = VideoAdapter(video_kwargs)
            self.va.connect(device)
            print(f"[CAM] 连接成功，开始取流。")
        except Exception as e:
            print(f"[CAM] 连接失败: {e}")
            return

        # 根据 max_fps 算出一个最小时间间隔（秒）
        min_interval = 1.0 / self.max_fps
        last_frame_time = 0

        try:
            while self.running.is_set():
                now = time.time()
                
                # 核心改进 2：时间闸门
                # 如果距离上次成功取图的时间太短，说明底层在吐重复的旧图缓存，直接跳过
                if now - last_frame_time < min_interval:
                    time.sleep(0.002)  # 睡 2ms，防 CPU 100% 空转
                    continue

                # 尝试取图
                rgb_image = self.va.get_ndarray(frame_format='rgb24')

                if rgb_image is not None:
                    # 更新上次取图的时间
                    last_frame_time = now
                    host_ts_ns = time.time_ns()
                    frame_data = (rgb_image, host_ts_ns)

                    # 入队逻辑：满则替换（抛弃旧帧）
                    try:
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()  # 扔掉旧的
                        except queue.Empty:
                            pass
                        try:
                            self.frame_queue.put_nowait(frame_data)  # 塞入新的
                        except queue.Full:
                            pass
                else:
                    # 如果没有拿到图像（极少情况），稍微歇一下
                    time.sleep(0.005)

        finally:
            # 清理连接
            if self.va:
                try:
                    self.va.disconnect()
                except Exception:
                    pass
            print(f"[CAM] 相机 {self.camera_id} 线程已退出。")

    def get_latest_frame(self, timeout=0.01):
        """供主循环调用：从队列获取最新帧，超时返回 (None, None)"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def stop(self):
        """通知后台线程停止"""
        self.running.clear()


class SensorStreamer(threading.Thread):
    def __init__(self, sensor_type, phone_ip="192.168.1.145", port=8080):
        """
        通用传感器取流线程。
        :param sensor_type: 传感器类型，例如 "android.sensor.accelerometer"
        :param phone_ip: 手机 IP 地址
        :param port: Sensor Server 端口
        """
        super().__init__(daemon=True)
        self.sensor_type = sensor_type
        self.url = f"ws://{phone_ip}:{port}/sensor/connect?type={sensor_type}"

        # 队列长度为 1：主循环没取走旧数据时，新数据会自动顶替它，绝不积压
        self.data_queue = queue.Queue(maxsize=1)
        
        self.running = threading.Event()
        self.running.set()
        self.ws = None

    def run(self):
        """后台线程：死循环接收 WebSocket 数据，打时间戳，塞进队列。"""
        # 从 URL 中提取一个简短的名字用于打印，例如 "accelerometer"
        short_name = self.sensor_type.split('.')[-1]
        print(f"[Sensor] 正在连接 {short_name}...")

        def on_message(ws, message):
            try:
                d = json.loads(message)
                phone_ts_ns = int(d["timestamp"])  # 手机硬件纳秒时间戳
                values = d["values"]
                if len(values) < 3:
                    return
                host_ts_ns = time.time_ns()        # 电脑接收时的纳秒时间戳

                # 数据结构：(三轴数值, 手机时间戳, 电脑接收时间戳)
                data = ([float(values[0]), float(values[1]), float(values[2])], 
                        phone_ts_ns, host_ts_ns)

                # 核心入队逻辑：满则替换
                try:
                    self.data_queue.put_nowait(data)
                except queue.Full:
                    try:
                        self.data_queue.get_nowait()  # 扔掉旧的
                    except queue.Empty:
                        pass
                    try:
                        self.data_queue.put_nowait(data)  # 塞入新的
                    except queue.Full:
                        pass
            except Exception as e:
                print(f"[Sensor] {short_name} 解析错误: {e}")

        def on_error(ws, error):
            # 只有在运行时发生错误才打印，关闭时断开不报错
            if self.running.is_set():
                print(f"[Sensor] {short_name} WebSocket 错误: {error}")

        def on_open(ws):
            print(f"[Sensor] {short_name} 连接成功，开始接收数据。")

        # 启动 WebSocket
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=on_message,
            on_error=on_error,
            on_open=on_open
        )
        self.ws.run_forever()

    def get_latest_data(self, timeout=0.01):
        """
        供主循环调用：从队列获取最新数据，超时返回 (None, None, None)。
        返回结构: (values_list, phone_ts_ns, host_ts_ns)
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None, None

    def stop(self):
        """通知后台线程停止，并关闭 WebSocket 连接"""
        self.running.clear()
        if self.ws:
            self.ws.close()



class LocationStreamer(threading.Thread):
    def __init__(self, poll_interval=1.0):
        super().__init__(daemon=True)
        self.poll_interval = poll_interval
        self.data_queue = queue.Queue(maxsize=1)
        self.running = threading.Event()
        self.running.set()
        
        # 核心修复：允许 * 号 (用于匹配脱敏的坐标)，同时兼容正常数字
        # 匹配格式例如：Location[gps 32.02****,118.61**** hAcc=...]
        self.pattern = re.compile(r'Location\[(gps|network|fused)\s+([\d\.\*\-]+),([\d\.\*\-]+)')
        
    def run(self):
        print("[GPS] 正在启动定位获取线程...")
        while self.running.is_set():
            try:
                # 强制 utf-8，并忽略无法识别的字符，防止崩溃
                output = subprocess.check_output(
                    ["adb", "shell", "dumpsys", "location"], 
                    text=True, 
                    encoding="utf-8",
                    errors="ignore",
                    stderr=subprocess.DEVNULL
                )
                
                # 从后往前找，获取最新的有效定位
                lines = output.splitlines()
                found = False
                for line in reversed(lines):
                    if "Location[" in line:
                        match = self.pattern.search(line)
                        if match:
                            source = match.group(1) # gps / network
                            lat_str = match.group(2)
                            lon_str = match.group(3)
                            
                            # 把星号去掉，方便有时候做简单的数值处理 (可选)
                            # 如果你需要保留星号，就不做 replace
                            lat_clean = lat_str.replace('*', '')
                            lon_clean = lon_str.replace('*', '')
                            
                            host_ts_ns = time.time_ns()
                            
                            data = (source, lat_str, lon_str, host_ts_ns)
                            try:
                                self.data_queue.put_nowait(data)
                            except queue.Full:
                                try:
                                    self.data_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                self.data_queue.put_nowait(data)
                            found = True
                            break
                
                if not found:
                    pass
                    
            except Exception as e:
                print(f"[GPS] 轮询出错: {e}")
                
            time.sleep(self.poll_interval)
            
    def get_latest_location(self, timeout=0.01):
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None, None, None

    def stop(self):
        self.running.clear()


def update_fps(frame_count, start_time, interval=1.0):
    """
    计算实时 FPS。
    
    :param frame_count: 当前累计的帧数
    :param start_time: 上一次计算 FPS 时的时间戳 (time.time())
    :param interval: 统计间隔（秒），默认 1 秒计算一次
    :return: (更新后的 frame_count, 更新后的 start_time, fps值)
             fps 值为 None 表示还没满 1 秒（无需更新显示）
             如果满 1 秒，fps 值为计算出的浮点数结果
    """
    frame_count += 1
    now = time.time()
    
    if now - start_time >= interval:
        fps = frame_count / (now - start_time)
        # 满 1 秒了，重置计数和起始时间，返回计算出的 fps
        return 0, now, fps
    else:
        # 还没满 1 秒，继续累积，返回 None 表示还没到计算时刻
        return frame_count, start_time, None


def main():
    # 1. 实例化并启动所有线程
    cam = CameraStreamer(camera_id=0, camera_size="1920x1080")
    
    phone_ip = "192.168.1.145"
    acc_sensor = SensorStreamer("android.sensor.accelerometer", phone_ip=phone_ip)
    gyro_sensor = SensorStreamer("android.sensor.gyroscope", phone_ip=phone_ip)
    mag_sensor = SensorStreamer("android.sensor.magnetic_field", phone_ip=phone_ip)
    
    gps_sensor = LocationStreamer(poll_interval=1.0)

    cam.start()
    acc_sensor.start()
    gyro_sensor.start()
    mag_sensor.start()
    gps_sensor.start()

    fps_frame_count = 0
    fps_start_time = time.time()
    current_fps = None
    
    latest_data = {"acc": None, "gyro": None, "mag": None, "gps": None}

    print("主循环开始，按 'q' 退出...")
    print("-" * 60)

    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Stream", 1280, 720)
    
    try:
        while True:
            # 2. 获取最新帧
            rgb_image, host_ts_ns = cam.get_latest_frame(timeout=0.01)

            # 3. 获取传感器数据 (非阻塞)
            acc_vals, _, _ = acc_sensor.get_latest_data(timeout=0.0)
            if acc_vals is not None: latest_data["acc"] = acc_vals
                
            gyro_vals, _, _ = gyro_sensor.get_latest_data(timeout=0.0)
            if gyro_vals is not None: latest_data["gyro"] = gyro_vals
                
            mag_vals, _, _ = mag_sensor.get_latest_data(timeout=0.0)
            if mag_vals is not None: latest_data["mag"] = mag_vals

            gps_src, gps_lat, gps_lon, gps_ts = gps_sensor.get_latest_location(timeout=0.0)
            if gps_lat is not None:
                latest_data["gps"] = (gps_src, gps_lat, gps_lon)

            # 4. 绘制与显示
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
                # ------------------------------------

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

            # 5. 处理退出
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