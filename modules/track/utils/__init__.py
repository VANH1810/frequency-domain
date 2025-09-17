import os
import sys
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
DATA = ROOT / 'data'
TRACKER_PATH = ROOT / "tracker"
EXAMPLES = ROOT / "tracking"
TRACKER_CONFIGS = ROOT / "traker" / "configs"
WEIGHTS = ROOT / "tracking" / "weights"
REQUIREMENTS = ROOT / "requirements.txt"

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))


# global logger
from loguru import logger

logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")
