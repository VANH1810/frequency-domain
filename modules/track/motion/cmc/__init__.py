from modules.track.motion.cmc.ecc import ECC
from modules.track.motion.cmc.orb import ORB
from modules.track.motion.cmc.sift import SIFT
from modules.track.motion.cmc.sof import SOF


def get_cmc_method(cmc_method):
    if cmc_method == 'ecc':
        return ECC
    elif cmc_method == 'orb':
        return ORB
    elif cmc_method == 'sof':
        return SOF
    elif cmc_method == 'sift':
        return SIFT
    else:
        return None
