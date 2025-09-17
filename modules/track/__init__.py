__version__ = '11.0.6'

from modules.track.postprocessing.gsi import gsi
from modules.track.trackers.deepocsort.deepocsort import DeepOcSort
from modules.track.trackers.ocsort.ocsort import OcSort

TRACKERS = ['ocsort', 'deepocsort']

__all__ = ("__version__",
           "OcSort","DeepOcSort",
           "create_tracker", "get_tracker_config", "gsi")
