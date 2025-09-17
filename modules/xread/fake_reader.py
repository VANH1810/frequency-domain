from modules.common.logger import get_app_logger
import numpy as np
import asyncio


class FakeReader:
    def __init__(self, camera_name, width, height, target_fps=30, loop=True):
        """
        Class for generating black frames of a given size at a fixed frame rate.
        :param width: Width of the generated frame.
        :param height: Height of the generated frame.
        :param target_fps: Target FPS (default: 30).
        :param loop: Whether to continuously generate frames (default: True).
        """
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.loop = loop
        self.is_opened = False
        self.logger = get_app_logger(camera_name, __name__)

    def open(self):
        """Simulate opening a video stream."""
        self.is_opened = True
        self.logger.info(f"✅ FakeReader initialized with size ({self.width}, {self.height}) at {self.target_fps} FPS.")
        return True

    async def read_frames(self):
        """Generate black frames at the specified FPS."""
        if not self.is_opened:
            self.logger.warning("⚠️ FakeReader is not opened!")
            return

        frame_delay = 1 / self.target_fps  # Adjust delay based on FPS
        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # Create a black frame

        while self.is_opened:
            yield black_frame  # Yield the generated frame
            await asyncio.sleep(frame_delay)  # Sleep to match FPS
            
            if not self.loop:
                self.logger.info("🎬 FakeReader has stopped generating frames.")
                break

    def release(self):
        """Simulate releasing the video resource."""
        self.is_opened = False
        self.logger.info("🔄 FakeReader resource released.")
