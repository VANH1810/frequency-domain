import cv2
import redis
import pickle
import threading
import time
from modules.common.logger import get_app_logger
from modules.common.timer import Timer

class RedisClient:
    def __init__(self, camera_name, host='localhost', 
                 port=6379, db=0, retry_delay=2, max_frames=200, 
                 max_retries=5, logger=None):
        """
        Initialize RedisClient with auto-reconnect capability.
        """
        self.camera_name = camera_name
        self.host = host
        self.port = port
        self.db = db
        self.retry_delay = retry_delay
        self.max_frames = max_frames
        self.max_retries = max_retries 
        self.redis_client = None
        self.lock = threading.Lock()
        self.running = True
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_app_logger(self.camera_name, __name__)

        self.connection_thread = threading.Thread(target=self._monitor_connection, daemon=True)
        self.connection_thread.start()

    def _connect(self):
        """
        Attempt to establish a Redis connection with limited retries.
        """
        retries = 0
        while self.running and retries < self.max_retries:
            try:
                client = redis.Redis(host=self.host, port=self.port, db=self.db, socket_timeout=5)
                client.ping()
                self.logger.info(f"[✔] Successfully connected to Redis from {self.camera_name}")
                return client
            except redis.ConnectionError:
                retries += 1
                self.logger.error(f"[⚠] {self.camera_name} redis connection failed. Retrying {retries}/{self.max_retries}...")
                time.sleep(self.retry_delay)
        self.logger.error(f"[❌] {self.camera_name} redis connection failed after max retries.")
        return None

    def _monitor_connection(self):
        """
        Background thread to ensure Redis connection remains active.
        """
        while self.running:
            if self.redis_client is None or not self._is_connected():
                self.logger.warning(f"[⚠] {self.camera_name} redis disconnected. Attempting to reconnect...")
                self.redis_client = self._connect()
            time.sleep(self.retry_delay)

    def _is_connected(self):
        """
        Check if Redis connection is alive.
        """
        try:
            return self.redis_client and self.redis_client.ping()
        except redis.ConnectionError:
            return False

    @Timer("redis push frame")
    def push_frame(self, frame_id, frame):
        """
        Push a frame to Redis and maintain only 'max_frames' per camera.
        """
        if not self._is_connected():
            self.logger.error(f"[⚠] {self.camera_name} redis not connected. Cannot push frame {frame_id}.")
            return

        with self.lock:
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = pickle.dumps(buffer)
                redis_key = f"camera:{self.camera_name}:frame:{frame_id}"

                pipeline = self.redis_client.pipeline()
                pipeline.set(redis_key, frame_data)

                oldest_frame_id = frame_id - self.max_frames
                if oldest_frame_id >= 0:
                    pipeline.delete(f"camera:{self.camera_name}:frame:{oldest_frame_id}")

                pipeline.execute()
            except Exception as e:
                self.logger.error(f"[⚠] {self.camera_name} error while pushing frame {frame_id}: {e}")
                self.redis_client = None

    # @Timer("redis get frame")
    def get_frame(self, frame_id):
        """
        Retrieve a frame from Redis by frame_id.
        """
        if not self._is_connected():
            self.logger.error(f"[⚠] {self.camera_name} redis not connected. Cannot retrieve frame {frame_id}.")
            return None

        with self.lock:
            try:
                redis_key = f"camera:{self.camera_name}:frame:{frame_id}"
                frame_data = self.redis_client.get(redis_key)
                return pickle.loads(frame_data) if frame_data else None
            except Exception as e:
                self.logger.error(f"[⚠] {self.camera_name} error while retrieving frame {frame_id}: {e}")
                self.redis_client = None
                return None

    def stop(self):
        """
        Stop the background connection monitoring thread and close Redis connection.
        """
        self.running = False
        if self.redis_client:
            self.redis_client.close()
            self.logger.info(f"[✔] {self.camera_name} redis connection closed.")
