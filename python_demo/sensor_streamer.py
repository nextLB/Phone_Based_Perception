


import queue
import threading
import time
import websocket
import json





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
