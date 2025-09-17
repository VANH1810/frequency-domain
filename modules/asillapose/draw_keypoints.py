import numpy as np
import cv2
from modules.common.constants import *


def draw_crowdpose_keypoints(image, bboxes=None, keypoints=None, min_conf=0.5, transparent=False, transparent_weight=0.5):
    return draw_custom_keypoints(image, bboxes, keypoints, CROWDPOSE_SKELETON, VIS_COLORS, min_conf, transparent, transparent_weight)

def draw_coco_keypoints(image, bboxes=None, keypoints=None, min_conf=0.5, transparent=False, transparent_weight=0.5):
    return draw_custom_keypoints(image, bboxes, keypoints, COCO_SKELETON, VIS_COLORS, min_conf, transparent, transparent_weight)

def draw_asilla_keypoints(image, bboxes=None, keypoints=None, min_conf=0.5, transparent=False, transparent_weight=0.5):
    return draw_custom_keypoints(image,  bboxes, keypoints, ASILLA_SKELETON, VIS_COLORS, min_conf, transparent, transparent_weight)

def draw_custom_keypoints(image, bboxes, keypoints, skeleton, colors=VIS_COLORS, min_conf=0.5, transparent=False, transparent_weight=0.5):
    """
    @brief Draw keypoints and skeleton on the image
    @param img Image array (OpenCV format)
    @param keypoints List of keypoints in [x, y, visibility] format
    @return Image with keypoints and skeleton drawn
    """
    if transparent:
        org_img = image.copy()
    box_thinkness= max(round(0.002 * min(image.shape[0], image.shape[1])), 1)
    kps_thinkness= max(round(0.006 * min(image.shape[0], image.shape[1])), 2)
    radius_circle = kps_thinkness + 2
    # Draw boxes
    if bboxes is not None:
        bboxes = np.array(bboxes)
        for box in bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, 
                            (int(x1), int(y1)), (int(x2), int(y2)), 
                            (np.random.randint(0, 255),  np.random.randint(0, 255),  np.random.randint(0, 255)), 
                            box_thinkness)
    # Draw keypoints
    if keypoints is not None:
        keypoints = np.array(keypoints)
        for i in range(len(keypoints)):
            # Draw keypoints and skeletons
            kp_length = keypoints.shape[1]
            if skeleton is not None:
                for ic, pair in enumerate(skeleton):
                    if pair[0] >= kp_length or pair[1] >= kp_length:
                        continue
                    kp1 = keypoints[i, pair[0]]
                    kp2 = keypoints[i, pair[1]]
                    if len(kp1) == 3 and len(kp2) == 3:
                        # Draw keypoints only if their scores exceed the threshold
                        if kp1[2] > min_conf and  kp2[2] > min_conf:
                            cv2.line(image, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), colors[ic], kps_thinkness)
                    else:
                        cv2.line(image, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), colors[ic], kps_thinkness)
            for ic, kp in enumerate(keypoints[i]):
                if len(kp) == 3:
                    if kp[2] > min_conf:
                        cv2.circle(image, (int(kp[0]), int(kp[1])), radius_circle, colors[ic], -1)
                else:
                    cv2.circle(image, (int(kp[0]), int(kp[1])), radius_circle, colors[ic], -1)  # Red color for keypoints  
                    
    if transparent:
        image = cv2.addWeighted(org_img, 1-transparent_weight, image, transparent_weight, 0)
    return image
