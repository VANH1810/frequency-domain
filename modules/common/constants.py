import numpy as np
from enum import Enum


MAX_SEQ_LEN = 60
MIN_BOX= 12
MAX_NAME_LENGTH = 25
PADDING_VAL = 114
ASILLA_SIGMAS = np.array([
    0.026,  # nose
    0.073,  # neck
    0.079,  # right shoulder
    0.072,  # rightelbow
    0.062,  # right wrist
    0.079,  # left shoulder
    0.072,  # left elbow
    0.062,  # left wrist
    0.107,  # right hip
    0.087,  # right knee
    0.089,  # right ankle
    0.107,  # left hip
    0.087,  # left knee
    0.089,  # left ankle
    0.025,  # righteye
    0.035,  # right ear
    0.025,  # left eye
    0.035,  # left ear
])
COCO_SIGMAS         = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
COCO_EASY_SIGMAS    = np.array([.26*2, .25*2, .25*2, .35*2, .35*2, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
CROWDPOSE_SKELETON  = np.array([[13, 14], [14, 1], [14, 2], 
                               [7, 8], [1, 3], [3, 5], 
                               [2, 4], [4, 6], [1, 7],
                               [2, 8], [7, 9], [9, 11], 
                               [8, 10], [10, 12]]) - 1
COCO_SKELETON = np.array([
    (0, 1), (0, 2), (1, 3), (2, 2), (2, 4), # Nose to right/left eyes and ears
    (0, 5), (5, 7), (7, 9),                 # Left arm
    (0, 6), (6, 8), (8, 10),                # Right leg
    (6, 12), (12, 14), (14, 16),            # Body connections (shoulders to hips)
    (5, 11), (11, 13), (13, 15),            # Right arm
    (12, 11), (6, 5)                        # Left leg (ankle to knee)
])


ASILLA_SKELETON = np.array([[0, 0], [0, 1],
                            [1, 2], [2, 3],
                            [3, 4], [1, 5],
                            [5, 6], [6, 7],
                            [8, 11], [2, 8],
                            [8, 9], [9, 10],
                            [5, 11], [11, 12],
                            [12, 13], [0, 14],
                            [14, 15], [0, 16],
                            [16, 17]])


VIS_COLORS = [(0, 215, 255), 
              (0, 255, 204), 
              (0, 134, 255), 
              (0, 255, 50),
              (77,255,222), 
              (77,196,255), 
              (77,135,255), 
              (191,255,77),  
              (77,255,77), 
              (77,222,255), 
              (255,156,127), 
              (0,127,255),  
              (255,127,77), 
              (0,77,255), 
              (127,127,255), 
              (255,0,127),  
              (0,127,0), 
              (255,255,128), 
              (0,0 ,50), 
              (0,150 ,50), 
              (255,180,20), 
              (20,180,255)]
 
class GCNSubclass(Enum):
    STANDING_STILL = 0
    SITTING_STILL = 1
    WALKING = 2
    FALLING_DOWN = 3
    LYING = 4
    STAGGERING = 5