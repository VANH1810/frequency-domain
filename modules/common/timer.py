import time
import functools
import threading
from collections import defaultdict, deque
from colorama import Fore, Style

global_times = defaultdict(lambda: deque(maxlen=15))
global_times_lock = threading.Lock()

class Timer:
    def __init__(self, message=None, time_key=None):
        self.message = message
        self.time_key = time_key  # detect_time, pose_time, track_time, vote_time, vis_time

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(instance, *args, **kwargs):
            start = time.perf_counter()
            result = func(instance, *args, **kwargs)
            end = time.perf_counter()
            duration_ms = (end - start) * 1000

            msg = self.message or func.__name__
            frame_number = None

            # Try to extract frame_number from args or kwargs
            if args:
                arg0 = args[0]
                if isinstance(arg0, dict) and "frame_number" in arg0:
                    frame_number = arg0["frame_number"]
            if "input_data" in kwargs and isinstance(kwargs["input_data"], dict):
                frame_number = kwargs["input_data"].get("frame_number", frame_number)

            if frame_number is not None:
                msg = f"[F{frame_number}] {msg}"

            log_str = f"{msg} take ~{duration_ms:.2f} ms"
            logger = getattr(instance, 'logger', None)

            # Highlight based on thresholds
            if duration_ms > 200:
                log_str += " ⚠️ SLOW"
                log_str_colored = Fore.YELLOW + log_str + Style.RESET_ALL
                if logger:
                    logger.warning(log_str)
                else:
                    print(log_str_colored)
            elif 180 <= duration_ms <= 200:
                log_str += " ⏱ NEAR SLOW"
                log_str_colored = Fore.CYAN + log_str + Style.RESET_ALL
                if logger:
                    logger.info(log_str)
                else:
                    print(log_str_colored)
            else:
                if logger:
                    logger.info(log_str)
                else:
                    print(log_str)

            # Store duration
            if self.time_key:
                with global_times_lock:
                    global_times[self.time_key].append(duration_ms)

            return result
        return wrapper


def get_max_avg_time():
    with global_times_lock:
        if not global_times:
            return None, None

        avg_times = {
            key: sum(times) / len(times)
            for key, times in global_times.items() if times
        }

        if not avg_times:
            return None, None

        max_key = max(avg_times, key=avg_times.get)
        return max_key, avg_times[max_key]
