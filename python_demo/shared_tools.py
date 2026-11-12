
import time


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



