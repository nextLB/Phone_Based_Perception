import queue
import threading
import time
import numpy as np
import re
import subprocess

from adbutils import adb
from mysc.core.video import VideoAdapter, VideoKwargs, CameraKwargs



# ---- 各相机完整可用尺寸（去重，来自 API2 availableStreamConfigurations）----
# 主摄节点（camera_id=0）：
# 4096x3072,4096x2304,4096x1864,3840x2160,3264x2448,3264x1836,3072x3072,
# 2760x1256,2648x1200,2560x1920,2560x1440,2400x1350,2400x1072,2384x1080,
# 2340x1080,2304x1728,2280x1080,2136x1200,2048x1536,1920x1440,1920x1080,
# 1600x1200,1600x720,1560x720,1520x720,1440x1440,1440x1080,1440x720,
# 1280x960,1280x720,1152x720,1088x1088,1080x720,960x720,800x600,800x400,
# 720x720,720x480,640x480,640x360,352x288,320x240,176x144
SIZES_MAIN = (
    "4096x3072,4096x2304,4096x1864,3840x2160,3264x2448,3264x1836,"
    "3072x3072,2760x1256,2648x1200,2560x1920,2560x1440,2400x1350,"
    "2400x1072,2384x1080,2340x1080,2304x1728,2280x1080,2136x1200,"
    "2048x1536,1920x1440,1920x1080,1600x1200,1600x720,1560x720,"
    "1520x720,1440x1440,1440x1080,1440x720,1280x960,1280x720,"
    "1152x720,1088x1088,1080x720,960x720,800x600,800x400,720x720,"
    "720x480,640x480,640x360,352x288,320x240,176x144"
)

# 前置 / 主摄第二节点（camera_id=1、3）比主摄多 4096x2048、2152x1212
SIZES_FRONT = (
    "4096x3072,4096x2304,4096x2048,4096x1864,3840x2160,3264x2448,"
    "3264x1836,3072x3072,2760x1256,2648x1200,2560x1920,2560x1440,"
    "2400x1350,2400x1072,2384x1080,2340x1080,2304x1728,2280x1080,"
    "2152x1212,2136x1200,1920x1440,1920x1080,1600x1200,1600x720,"
    "1560x720,1520x720,1440x1440,1440x1080,1440x720,1280x960,"
    "1280x720,1152x720,1088x1088,1080x720,960x720,800x600,800x400,"
    "720x720,720x480,640x480,640x360,352x288,320x240,176x144"
)

# 超广角（camera_id=4）最大 3264x2448，且有独有的 3264x1488、2448x2448、1632x1224
SIZES_UW = (
    "3264x2448,3264x1836,3264x1488,2760x1256,2648x1200,2560x1920,"
    "2560x1440,2448x2448,2400x1350,2400x1072,2384x1080,2340x1080,"
    "2304x1728,2280x1080,2136x1200,1920x1440,1920x1080,1632x1224,"
    "1600x1200,1600x720,1560x720,1520x720,1440x1440,1440x1080,"
    "1440x720,1280x960,1280x720,1152x720,1088x1088,1080x720,"
    "960x720,800x600,800x400,720x720,720x480,640x480,640x360,"
    "352x288,320x240,176x144"
)


# ---- 每个相机的元信息 ----
CAMERAS = {
    0: {
        "name": "后置主摄",
        "hal_device": 0,
        "facing": "后置",
        "aperture": "f/1.8",
        "focal_length": "4.84mm",
        "flash": True,
        # 切换相机时使用的默认尺寸。你原来验证过 4096x3072 可用：
        # 想回到全尺寸，把这里改成 "4096x3072" 即可（帧率可能不满 30，注意发热）。
        "default_size": "1920x1080",
        "max_size": "4096x3072",
        "recommended": ["1920x1080", "2560x1440", "3840x2160", "4096x3072"],
        "sizes": SIZES_MAIN,
        "desc": "日常主力，视角广、画质最好；全尺寸 4096x3072 已实测可用",
    },
    1: {
        "name": "前置自拍",
        "hal_device": 1,
        "facing": "前置",
        "aperture": "f/2.0",
        "focal_length": "3.2mm",
        "flash": False,
        "default_size": "1920x1080",
        "max_size": "4096x3072",
        "recommended": ["1280x720", "1920x1080", "2560x1440", "4096x3072"],
        "sizes": SIZES_FRONT,
        "desc": "自拍镜头，支持 AF；画面可能左右镜像（取决于 HAL 默认）",
    },
    3: {
        "name": "后置主摄-第二节点",
        "hal_device": 2,
        "facing": "后置",
        "aperture": "f/1.8",
        "focal_length": "4.84mm",
        "flash": True,
        "default_size": "1920x1080",
        "max_size": "4096x3072",
        "recommended": ["1920x1080", "2560x1440", "3840x2160", "4096x3072"],
        "sizes": SIZES_FRONT,
        "desc": "参数与 camera_id=0 几乎相同，OPPO 拆出的第二个主摄 HAL 节点，"
                "多用于高像素 / 多帧 / 特殊算法；预览画面通常与 0 相近",
    },
    4: {
        "name": "后置超广角",
        "hal_device": 3,
        "facing": "后置",
        "aperture": "f/2.2",
        "focal_length": "1.6mm",
        "flash": True,
        "default_size": "1920x1080",
        "max_size": "3264x2448",
        "recommended": ["1280x720", "1920x1080", "2560x1440", "3264x2448"],
        "sizes": SIZES_UW,
        "desc": "116° 超广角，视角明显更宽；最大只到 3264x2448",
    },
}

# 键盘数字 -> camera_id（公开 ID 是 0/1/3/4，没有 2）
KEY_TO_CAMERA = {
    ord('1'): 0,
    ord('2'): 1,
    ord('3'): 3,
    ord('4'): 4,
}


# 公开 camera_id -> dumpsys 里的 "device@1.1/internal/N"
_PUBLIC_TO_DEVICE = {0: 0, 1: 1, 3: 2, 4: 3}



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

    @staticmethod
    def get_camera_calibration_from_file(dump_path, camera_id=0):
        """从 dumpsys media.camera 导出文本中解析相机标定参数。

        返回 dict: K / raw_intrinsic / pose_rotation / pose_translation / active_array，
        失败返回 None。自动识别 UTF-8 / UTF-16 LE / UTF-16 BE（兼容 PowerShell 重定向）。
        """
        device_id = _PUBLIC_TO_DEVICE.get(camera_id, camera_id)

        # ---- 读文件：二进制 + BOM 检测 ----
        with open(dump_path, "rb") as f:
            raw = f.read()
        if raw.startswith(b"\xff\xfe"):
            content = raw[2:].decode("utf-16-le", errors="ignore")
        elif raw.startswith(b"\xfe\xff"):
            content = raw[2:].decode("utf-16-be", errors="ignore")
        elif raw.startswith(b"\xef\xbb\xbf"):
            content = raw.decode("utf-8-sig", errors="ignore")
        else:
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = raw.decode("utf-16-le", errors="ignore")

        lines = content.splitlines(keepends=True)
        dev_kw, sec_kw = f"device@1.1/internal/{device_id}", "static information"

        # ---- 定位段落：标题行含 device_id + static information（容错：允许拆行） ----
        start = None
        for i, line in enumerate(lines):
            if dev_kw in line and any(sec_kw in lines[j] for j in range(i, min(i + 4, len(lines)))):
                start = i
                break
        if start is None:
            print(f"[CAL] 未找到 device {device_id} 的 static information 段落")
            return None

        # 结束行 = 下一个 device 的 static information 行（跳过中间的 dumpState 段）
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if "device@1.1/internal/" in lines[i] and sec_kw in lines[i]:
                end = i
                break
        seg = "".join(lines[start:end])

        # ---- 字段解析 ----
        def parse_float_array(name, count):
            m = re.search(
                rf"\b{name}\b[^\n]*float\[{count}\]\s*\[\s*([^\]]+?)\s*\]",
                seg, re.DOTALL
            )
            if not m:
                return None
            vals = [float(x) for x in m.group(1).split()]
            return vals if len(vals) == count else None

        intr = parse_float_array("intrinsicCalibration", 5)
        if intr is None:
            print(f"[CAL] device {device_id} 段落内未解析到 intrinsicCalibration")
            return None
        rot   = parse_float_array("poseRotation", 4)
        trans = parse_float_array("poseTranslation", 3)

        active = None
        m = re.search(
            r"android\.sensor\.info\.activeArraySize[^\n]*int32\[4\]\s*\[\s*([^\]]+?)\s*\]",
            seg, re.DOTALL
        )
        if m:
            v = [int(x) for x in m.group(1).split()]
            if len(v) == 4:
                active = (v[2] - v[0], v[3] - v[1])

        # ---- 组装 K（s 为偏斜，通常为 0） ----
        fx, fy, cx, cy, s = intr
        K = np.array([[fx, s, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float64)

        return {
            "K": K,
            "raw_intrinsic": intr,
            "pose_rotation": rot,
            "pose_translation": trans,
            "active_array": active,
        }


    @staticmethod
    def scale_intrinsics(K, ref_size, new_size):
        """按比例缩放 K。前提：new_size 与 ref_size 宽高比一致（纯缩放，无裁剪）。

        ref_size / new_size 均为 (W, H)。若要处理 4:3 → 16:9 的 center crop，
        需另写函数并在 cy 上减去裁剪偏移。
        """
        sx = new_size[0] / ref_size[0]
        sy = new_size[1] / ref_size[1]
        K_new = K.copy()
        K_new[0, 0] *= sx
        K_new[1, 1] *= sy
        K_new[0, 2] *= sx
        K_new[1, 2] *= sy
        return K_new

    