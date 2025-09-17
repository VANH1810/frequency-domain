import numpy as np
import cv2
from modules.common.constants import VIS_COLORS, ASILLA_SKELETON, GCNSubclass
from modules.common.timer import get_max_avg_time
import torch

def coco2asilla(coco_keypoints): 
    coco_keypoints = np.vstack([coco_keypoints, np.zeros((1, 3))])

    new_indices = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 4, 1, 3]
    keypoints = coco_keypoints[new_indices]  

    left_shoulder, right_shoulder = coco_keypoints[5], coco_keypoints[6]
    if left_shoulder[2] > 0 and right_shoulder[2] > 0: 
        keypoints[1, :2] = (left_shoulder[:2] + right_shoulder[:2]) / 2  
        keypoints[1, 2] = (left_shoulder[2] + right_shoulder[2]) / 2  
    
    return keypoints 


def draw_keypoint(image, kp_array, skeleton=None, colors=None, min_conf=0.1):
    kps_thickness = max(round(0.006 * min(image.shape[0], image.shape[1])), 2)
    radius_circle = kps_thickness + 2
    
    if colors is None:
        colors = VIS_COLORS
    if skeleton is None:
        skeleton = ASILLA_SKELETON
        
    if skeleton is not None:
        for ic, (idx1, idx2) in enumerate(skeleton):
            if idx1 >= len(kp_array) or idx2 >= len(kp_array):
                continue  
            kp1, kp2 = kp_array[idx1], kp_array[idx2]
            if kp1[2] > min_conf and kp2[2] > min_conf:
                cv2.line(image, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), 
                         colors[ic % len(colors)], kps_thickness)
    
    for ic, (x, y, conf) in enumerate(kp_array):
        if conf > min_conf:  
            cv2.circle(image, (int(x), int(y)), radius_circle, colors[ic % len(colors)], -1)
            
def draw_list_action_object(image, list_actObj, show_pose=False, show_box=False, show_pid=False, \
    skeleton=None, colors=None, transparent=False, transparent_weight=0.5, min_conf=0.1, fps_sys=None):
    if transparent:
        org_img = image.copy()

    box_thickness = max(round(0.002 * min(image.shape[0], image.shape[1])), 2)
    font_scale = 0.6
    font_thickness = 2
    text_padding = 5  

    if colors is None:
        colors = VIS_COLORS
    if skeleton is None:
        skeleton = ASILLA_SKELETON
    
    for actObj in list_actObj:
        track_obj = actObj.track_obj  
        pid = track_obj.pid  
        box = track_obj.box  
        kp = track_obj.kp  
        action = actObj.action  

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # draw bounding box if show_box=True
        if show_box:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

        # draw ID + Action if show_pid=True
        if show_pid:
            text = f"ID: {pid} - {action}"
            # label_action = GCNSubclass(action).name.lower() if action != 100 else "other"
            # text = f"ID: {pid} - {label_action}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_w, text_h = text_size

            # draw background text if show_box=True
            if show_box:
                text_bg_x1, text_bg_y1 = x1, y1 - text_h - text_padding * 2
                text_bg_x2, text_bg_y2 = x1 + text_w + text_padding * 2, y1

                cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)

            #  draw text on image
            cv2.putText(image, text, (x1 + text_padding, y1 - text_padding), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA)

        # draw pose if show_pose=True and keypoint is not None
        if show_pose and kp is not None and skeleton is not None:
            kp_array = np.array(kp)
            draw_keypoint(image, kp_array, skeleton, colors, min_conf)
            
    
    # 🟡 Draw FPS with styled background box
    if fps_sys is not None:
        fps_text = f"FPS: {fps_sys:.1f}"
        
        max_key, max_value = get_max_avg_time()
        if max_key is not None and max_value is not None:
            fps_sys_new = round(1000/max_value,2)
            fps_text = f"FPS: {fps_sys_new:.1f}"
        else:
            print("No timing data available.")
                
        fps_font_scale = 0.7
        fps_color = (0, 255, 255)
        fps_thickness = 2
        margin = 10

        text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, fps_thickness)
        text_w, text_h = text_size
        x, y = margin, margin + text_h

        # Draw semi-transparent background for FPS
        overlay = image.copy()
        cv2.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Draw the actual FPS text
        cv2.putText(image, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fps_font_scale, fps_color, fps_thickness, cv2.LINE_AA)
        
    if transparent:
        image = cv2.addWeighted(org_img, 1 - transparent_weight, image, transparent_weight, 0)

    return image


SYMMETRIC_PAIRS = [
    ((5, 7), (6, 8)),     # shoulder - elbow
    ((7, 9), (8, 10)),    # elbow - wrist
    ((11, 13), (12, 14)), # hip - knee
    ((13, 15), (14, 16)), # knee - ankle
]

RELATIVE_TO_NOSE = [
    (1, 2),  # left eye – right eye
    (3, 4),  # left ear – right ear
]

def filter_keypoints_relative_batch(
    kpts_batch: torch.Tensor,
    conf_thresh=0.1,
    ratio_thresh_limbs=2.5,
    ratio_thresh_face=1.5,
):
    """
    Lọc outlier keypoints dựa vào khoảng cách tương đối giữa các cặp đối xứng,
    bao gồm tay/chân và mặt (tai/mắt so với mũi).

    Args:
        kpts_batch: (B, 17, 3)
        conf_thresh: threshold confidence
        ratio_thresh_limbs: ngưỡng tỉ lệ cho tay/chân
        ratio_thresh_face: ngưỡng tỉ lệ cho tai/mắt so với mũi
    Returns:
        Tensor: (B, 17, 3) đã lọc
    """

    kpts_batch = kpts_batch.clone()
    B = kpts_batch.shape[0]

    # ==== Tay/chân ====
    for (left_pair, right_pair) in SYMMETRIC_PAIRS:
        l1, l2 = left_pair
        r1, r2 = right_pair

        l1_kpt, l2_kpt = kpts_batch[:, l1], kpts_batch[:, l2]
        r1_kpt, r2_kpt = kpts_batch[:, r1], kpts_batch[:, r2]

        l_valid = (l1_kpt[:, 2] > conf_thresh) & (l2_kpt[:, 2] > conf_thresh)
        r_valid = (r1_kpt[:, 2] > conf_thresh) & (r2_kpt[:, 2] > conf_thresh)

        left_dist = torch.norm(l1_kpt[:, :2] - l2_kpt[:, :2], dim=1)
        right_dist = torch.norm(r1_kpt[:, :2] - r2_kpt[:, :2], dim=1)

        both_valid = l_valid & r_valid
        ratio = left_dist / (right_dist + 1e-6)

        invalid_left = (ratio > ratio_thresh_limbs) & both_valid
        invalid_right = (ratio < 1 / ratio_thresh_limbs) & both_valid

        kpts_batch[invalid_left, l2] = 0.0
        kpts_batch[invalid_right, r2] = 0.0

    # ==== Tai/mắt so với mũi ====
    for left, right in RELATIVE_TO_NOSE:
        center = 0  # mũi

        nose_kpt = kpts_batch[:, center]
        left_kpt = kpts_batch[:, left]
        right_kpt = kpts_batch[:, right]

        nose_valid = nose_kpt[:, 2] > conf_thresh
        left_valid = left_kpt[:, 2] > conf_thresh
        right_valid = right_kpt[:, 2] > conf_thresh

        dist_left = torch.norm(nose_kpt[:, :2] - left_kpt[:, :2], dim=1)
        dist_right = torch.norm(nose_kpt[:, :2] - right_kpt[:, :2], dim=1)

        both_valid = nose_valid & left_valid & right_valid
        ratio = dist_left / (dist_right + 1e-6)

        invalid_left = (ratio > ratio_thresh_face) & both_valid
        invalid_right = (ratio < 1 / ratio_thresh_face) & both_valid

        kpts_batch[invalid_left, left] = 0.0
        kpts_batch[invalid_right, right] = 0.0

    return kpts_batch
