import queue
import threading
import time
import subprocess
import re






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




