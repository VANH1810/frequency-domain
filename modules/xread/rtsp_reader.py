from modules.common.logger import get_app_logger
import asyncio
import cv2


class RTSPReader:
    def __init__(self, camera_name, url, reconnect_delay=5, width=1280, height=720, backend="opencv", max_queue_size=5):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.backend = backend
        self.cap = None
        self.is_connected = False
        self.logger = get_app_logger(camera_name, __name__)
        self.width = width 
        self.height = height
        self.frame_queue = asyncio.Queue(maxsize=max_queue_size)
        self.stop_event = asyncio.Event()
        self.lock = asyncio.Lock()

    async def connect(self):
        """Open camera and launch internal read loop"""
        while not self.stop_event.is_set():
            try:
                if self.backend == "gstreamer":
                    self.cap = cv2.VideoCapture(
                        f"rtspsrc location={self.url} protocols=tcp ! decodebin ! nvvideoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=1",
                        cv2.CAP_GSTREAMER
                    )
                else:
                    self.cap = cv2.VideoCapture(self.url)

                if self.cap.isOpened():
                    self.is_connected = True
                    self.logger.info(f"Connected to RTSP stream using {self.backend}")
                    
                    # Launch background frame reader
                    asyncio.create_task(self._reader_loop())
                    return True

                else:
                    self.logger.warning("cv2.VideoCapture not opened")

            except Exception as e:
                self.logger.error(f"Failed to connect to RTSP stream: {e}")

            self.logger.info(f"Retrying connection in {self.reconnect_delay} seconds...")
            await asyncio.sleep(self.reconnect_delay)

        return False

    async def _reader_loop(self):
        """Internal loop to read from cap and push to queue"""
        while not self.stop_event.is_set() and self.cap and self.cap.isOpened():
            async with self.lock:
                ret, frame = self.cap.read()

            if not ret:
                self.logger.warning("Frame read failed, exiting read loop.")
                self.is_connected = False
                break

            frame = cv2.resize(frame, (self.width, self.height))

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            await self.frame_queue.put(frame)

            # Optional: Limit FPS here if needed
            await asyncio.sleep(0.01)  # ~100 FPS max

    async def read_frame(self):
        """Pop a frame from queue for external use"""
        try:
            frame = await self.frame_queue.get()
            return True, frame
        except Exception as e:
            self.logger.error(f"Failed to get frame from queue: {e}")
            return False, None

    def release(self):
        """Stop everything"""
        self.stop_event.set()
        if self.cap:
            self.cap.release()
