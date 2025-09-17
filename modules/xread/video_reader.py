from modules.common.logger import get_app_logger
import cv2
import asyncio


class VideoReader:
    def __init__(self, camera_name, video_path, target_fps=None, loop=True, reconnect_delay=5, width=1280, height=720, backend="opencv", max_queue_size=5):
        self.video_path = video_path
        self.target_fps = target_fps
        self.loop = loop
        self.reconnect_delay = reconnect_delay
        self.width = width
        self.height = height
        self.backend = backend
        self.logger = get_app_logger(camera_name, __name__)
        
        self.cap = None
        self.original_fps = None
        self.is_opened = False
        self.stop_event = asyncio.Event()
        
        self.frame_queue = asyncio.Queue(maxsize=max_queue_size)
        self.lock = asyncio.Lock()

    async def open(self):
        """Open the video file and start reader loop."""
        attempts = 0
        while not self.is_opened and attempts < 5:
            if self.stop_event.is_set():
                self.logger.info("🛑 Stop signal received. Exiting open() loop.")
                return False
            try:
                if self.backend == "gstreamer":
                    self.cap = cv2.VideoCapture(
                        f"filesrc location={self.video_path} ! decodebin ! nvvideoconvert ! video/x-raw,format=BGR ! appsink",
                        cv2.CAP_GSTREAMER)
                else:
                    self.cap = cv2.VideoCapture(self.video_path)

                if not self.cap.isOpened():
                    self.logger.error(f"❌ Unable to open video: {self.video_path}")
                    attempts += 1
                    await asyncio.sleep(self.reconnect_delay)
                    continue

                self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.target_fps is None:
                    self.target_fps = self.original_fps

                self.is_opened = True
                self.logger.info(f"✅ Video opened: {self.video_path}")
                self.logger.info(f"🎥 FPS: {self.original_fps:.2f}, Target FPS: {self.target_fps:.2f}, Resolution: {self.width}x{self.height}")
                
                # Start background reader
                asyncio.create_task(self._reader_loop())
                return True

            except Exception as e:
                self.logger.error(f"Error opening video: {e}")
                attempts += 1
                await asyncio.sleep(self.reconnect_delay)

        return False

    async def _reader_loop(self):
        """Loop to continuously read video and push frames to queue."""
        frame_delay = 1 / self.target_fps
        while not self.stop_event.is_set() and self.cap and self.cap.isOpened():
            async with self.lock:
                ret, frame = self.cap.read()

            if not ret:
                if self.loop:
                    self.logger.info("🔁 Video ended, restarting...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.logger.info("🎬 Video ended.")
                    self.is_opened = False
                    break

            frame = cv2.resize(frame, (self.width, self.height))

            # Drop oldest if full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            await self.frame_queue.put(frame)
            await asyncio.sleep(frame_delay)

        self.logger.info("📴 Exiting video read loop.")

    async def read_frame(self):
        """Retrieve a frame from the queue."""
        try:
            frame = await self.frame_queue.get()
            return True, frame
        except Exception as e:
            self.logger.error(f"❌ Failed to get frame from queue: {e}")
            return False, None

    def release(self):
        """Clean up resources."""
        self.stop_event.set()
        if self.cap:
            self.cap.release()
            self.cap = None
            self.logger.info("🎞️ Video capture released.")
