import cv2
import base64
import threading
import time
import json
import asyncio
import websockets
from collections import deque
from modules.common.logger import get_app_logger
from modules.common.config import get_app_config
from concurrent.futures import ThreadPoolExecutor

class StreamManager:
    def __init__(self, camera_name="cam1", timeout=60, queue_size=5):
        self.camera_name = camera_name
        self.timeout = timeout
        self.queue_size = queue_size
        self.camera_queues = {}
        self.camera_locks = {}
        self.client_last_activity = {}
        self.clients = set()
        self.logger = get_app_logger(self.camera_name, __name__)
        self.config = get_app_config(self.camera_name)

        self.port = self.config.getint("streamout", "port", fallback=9101)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.new_event_loop()
        self.server_thread = threading.Thread(target=self.start_server_loop)
        self.server_thread.start()

        self.cleanup_thread = threading.Thread(target=self.cleanup_inactive_clients, daemon=True)
        self.cleanup_thread.start()

    def start_server_loop(self):
        asyncio.set_event_loop(self.loop)
        start_server = websockets.serve(self.handler, "0.0.0.0", self.port)
        self.loop.run_until_complete(start_server)
        self.loop.run_forever()

    async def handler(self, websocket, path):
        client_id = id(websocket)
        self.logger.info(f"New client connected: {client_id}")
        self.clients.add(websocket)
        self.client_last_activity[client_id] = time.time()

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "heartbeat":
                        self.client_last_activity[client_id] = time.time()
                        self.logger.info(f"Received heartbeat from client {client_id}")
                    else:
                        self.logger.info(f"Unexpected message: {message}")
                except Exception as e:
                    self.logger.error(f"Error parsing message: {e}")
        except Exception as e:
            self.logger.info(f"Client {client_id} disconnected: {e}")
        finally:
            self.clients.discard(websocket)
            self.client_last_activity.pop(client_id, None)

    def stream_video(self, cam_id):
        if cam_id not in self.camera_queues:
            self.camera_queues[cam_id] = deque(maxlen=self.queue_size)
            self.camera_locks[cam_id] = threading.Lock()

    def cleanup_inactive_clients(self):
        while True:
            time.sleep(5)
            current_time = time.time()
            inactive_clients = [
                client_id for client_id, last_activity in self.client_last_activity.items()
                if current_time - last_activity > self.timeout
            ]
            for client_id in inactive_clients:
                self.logger.info(f"Cleaning up inactive client {client_id}")
                self.client_last_activity.pop(client_id, None)

    def start_video_stream(self, cam_ids):
        for cam_id in cam_ids:
            thread = threading.Thread(target=self.stream_video, args=(cam_id,))
            thread.start()

    def push_image(self, cam_id, image):
        self.executor.submit(self._process_and_push, cam_id, image)

    def _process_and_push(self, cam_id, image):
        if not self.clients:
            self.logger.warning(f"[{cam_id}] ❌ No clients connected. Skipping frame push.")
            return

        self.logger.info(f"[{cam_id}] ✅ Clients connected: {len(self.clients)}. Preparing to push frame...")

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        try:
            success, buffer = cv2.imencode('.jpg', image, encode_param)
            if not success:
                self.logger.error(f"[{cam_id}] ❌ Failed to encode image to JPEG.")
                return
        except Exception as e:
            self.logger.exception(f"[{cam_id}] ❌ Exception during image encoding: {e}")
            return

        try:
            jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self.logger.exception(f"[{cam_id}] ❌ Exception during base64 encoding: {e}")
            return

        message = {"cam_id": cam_id, "frame": jpg_as_base64}
        try:
            message_json = json.dumps(message)
        except Exception as e:
            self.logger.exception(f"[{cam_id}] ❌ Failed to serialize message to JSON: {e}")
            return

        self.logger.info(f"[{cam_id}] 📤 Frame prepared: size={len(jpg_as_base64)} characters")

        if cam_id not in self.camera_queues:
            self.camera_queues[cam_id] = deque(maxlen=self.queue_size)

        with self.camera_locks[cam_id]:
            self.camera_queues[cam_id].append(message_json)
            self.logger.debug(f"[{cam_id}] 🔁 Appended frame to queue (size now = {len(self.camera_queues[cam_id])})")

        # Send to all connected clients
        for client in list(self.clients):
            try:
                asyncio.run_coroutine_threadsafe(client.send(message_json), self.loop)
            except Exception as e:
                self.logger.error(f"[{cam_id}] ❌ Error sending frame to client: {e}")

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.server_thread.join()