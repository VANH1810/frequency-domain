import threading
import queue
from modules.common.logger import get_app_logger
from modules.common.config import get_app_config
import time 
from collections import defaultdict
import requests
import uuid

class AlertType:
    FALLING = 'Falling'
    STAGGERING = 'Staggering'
    LOITERING = 'Loitering'
    
class AlertID:
    FALLING = 0
    STAGGERING = 1
    LOITERING = 2
    
    
class AlertRecord:
    def __init__(self):
        self.last_alert_time = {}  
        self.last_seen_frame = 0
        self.last_seen_time = time.time()


class AlertManager:
    def __init__(self, camera_name, base_url=None, queue_maxsize=100): 
        self.camera_name = camera_name
        self.logger = get_app_logger(camera_name, __name__)  
        self.config_reader = get_app_config(camera_name)
        self.base_url = base_url 
        self.alert_mapping = {
            AlertType.FALLING: AlertID.FALLING,
            AlertType.STAGGERING: AlertID.STAGGERING,
            AlertType.LOITERING: AlertID.LOITERING
        }

        self.min_alert_interval_time = self.config_reader.getint("alert", "min_alert_interval_time", fallback=30)
        self.cleanup_time = self.config_reader.getint("alert", "cleanup_time", fallback=60) 
        self.cleanup_frame = self.config_reader.getint("alert", "cleanup_frame", fallback=80) 
        
        self.pid_records = defaultdict(AlertRecord)
        self.alert_queue = queue.Queue(maxsize=queue_maxsize)
        self.threads = []
        self.running = True
        
    def can_alert(self, pid, action):
        record = self.pid_records[pid]
        current_time = time.time()
        
        if action not in record.last_alert_time:
            return True
            
        time_since_last_alert = current_time - record.last_alert_time[action]
        return time_since_last_alert >= self.min_alert_interval_time

    def update_pid_record(self, pid, frame_number):
        record = self.pid_records[pid]
        record.last_seen_frame = frame_number
        record.last_seen_time = time.time()
        
    def cleanup_old_records(self, current_frame):
        current_time = time.time()
        inactive_pids = []
        
        for pid, record in self.pid_records.items():
            time_inactive = current_time - record.last_seen_time
            frames_absent = current_frame - record.last_seen_frame
            
            if (time_inactive >= self.cleanup_time or 
                frames_absent >= self.cleanup_frame):
                inactive_pids.append(pid)
        
        for pid in inactive_pids:
            del self.pid_records[pid]
            self.logger.info(f"🧹 Cleaned up inactive PID: {pid}")
            
    def send_alert_request(self, payload):
        try:
            response = requests.post(f"{self.base_url}/api/trigger_alert_record", json=payload, timeout=5)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"❌ Error sending alert: {e}")
            return None

    def trigger_api(self, frm_idx, pid, action, f_actobj):
        alert_id = str(uuid.uuid4())
        payload = {
            "alert_id": alert_id,
            "camera_name": self.camera_name,
            "frame_index": frm_idx,
            "pid": pid,
            "action": action,
            "f_actobj": f_actobj.to_dict()
        }
        self.alert_queue.put((pid, action, payload))
        
    def alert_worker(self):
        while self.running:
            try:
                pid, action, payload = self.alert_queue.get(timeout=1)
                response = self.send_alert_request(payload)
                if response:
                    self.logger.info(f"✅ Sent {action} - pid {pid} alert. Response: {response.json()}")
                    self.pid_records[pid].last_alert_time[action] = time.time()
                time.sleep(0.1)
            except queue.Empty:
                continue

    def start_sending_threads(self, num_worker=4):
        for _ in range(num_worker):
            thread = threading.Thread(target=self.alert_worker, daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join()
