import cv2 
import numpy as np
import torch
# from modules.common.timer import Timer

#======== mean,std for detection stage ==============
mean_det = [123.675, 116.28, 103.53]
std_det = [58.395, 57.12, 57.375]
input_det_w, input_det_h = 640, 640

#======== mean,std for pose estimate stage ==========
mean_pose = [123.675, 116.28, 103.53]
std_pose = [58.395, 57.12, 57.375]
num_points = 17
input_pose_w, input_pose_h = 288, 384
input_pose_w_mini, input_pose_h_mini = 192, 256

def resize_fast(image, target_w, target_h):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(scale * w), int(scale * h)
    
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    
    offset_x = (target_w - nw) // 2
    offset_y = (target_h - nh) // 2
    pad_image = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    pad_image[offset_y:offset_y + nh, offset_x:offset_x + nw] = resized
    
    return pad_image, offset_x, offset_y
    

#==============Fuctions SUPPORT DETECTION ==========
class Box:
    def __init__(self, x1, y1, x2, y2, cls, conf):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.cls, self.conf = cls, conf


def draw_boxes(image, boxes):
    for idx, b in enumerate(boxes):
        # print(b.x1, b.y1, b.x2, b.y2)
        cv2.rectangle(image, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{int(b.cls)}:{b.conf:.2f}", (int(b.x1), int(b.y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)        
    return image



# @Timer("rtmdet preprocess")
def preprocess_det_fast(img_bgr):
    """
    Resize, BGR→RGB, normalize, NCHW, float32  – vectorized, zero loops.
    Return: (tensor, (offset_x, offset_y))
    """
    # 1) letterbox / resize 
    resized, off_x, off_y = resize_fast(img_bgr, input_det_w, input_det_h)  # BGR HxWx3

    # 2) BGR→RGB  
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)   # HxWx3 float32

    # 3) normalize  (broadcasting)
    img = (img - mean_det) / std_det        # mean_det & std_det: shape (3,)

    # 4) HWC → CHW → add batch dim
    tensor = img.transpose(2, 0, 1)[None, ...].astype(np.float32)   # (1,3,H,W)

    return tensor, (off_x, off_y)

# @Timer("rtmdet nms")
def nms_fast(boxes, iou_threshold):
    if not boxes:
        return []
    
    n_boxes = len(boxes)
    coords = np.zeros((n_boxes, 4), dtype=np.float32)
    scores = np.zeros(n_boxes, dtype=np.float32)
    
    for i, box in enumerate(boxes):
        coords[i, 0] = box.x1
        coords[i, 1] = box.y1
        coords[i, 2] = box.x2
        coords[i, 3] = box.y2
        scores[i] = box.conf
    
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        
        if indices.size == 1:
            break
            
        indices = indices[1:]
        
        current_box = coords[current].reshape(1, 4)
        remaining_boxes = coords[indices]
        
        xx1 = np.maximum(current_box[0, 0], remaining_boxes[:, 0])
        yy1 = np.maximum(current_box[0, 1], remaining_boxes[:, 1])
        xx2 = np.minimum(current_box[0, 2], remaining_boxes[:, 2])
        yy2 = np.minimum(current_box[0, 3], remaining_boxes[:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        area_current = (current_box[0, 2] - current_box[0, 0]) * (current_box[0, 3] - current_box[0, 1])
        
        
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        
        
        union = area_current + area_remaining - intersection + 1e-6
        iou = intersection / union
        
        
        indices = indices[iou < iou_threshold]
    
    
    result = [boxes[i] for i in keep]
    return result

# @Timer("rtmdet postprocess")
def postprocess_det_fast(boxes_result, img_w, img_h, offset, conf_thre, iou_thre):
    result = boxes_result[0]
    valid_indices = np.where(result[:, 5] >= conf_thre)[0]
    
    if len(valid_indices) == 0:
        return []
    
    valid_boxes = result[valid_indices]
    
    boxes = []
    for i in range(len(valid_boxes)):
        buff = valid_boxes[i]
        box = Box(buff[0], buff[1], buff[2], buff[3], buff[4], buff[5])
        boxes.append(box)
    
    boxes = nms_fast(boxes, iou_thre)
    
    scale_x = img_w / (input_det_w - 2 * offset[0])
    scale_y = img_h / (input_det_h - 2 * offset[1])
    
    for b in boxes:
        b.x1 = max((b.x1 - offset[0]) * scale_x, 0)
        b.y1 = max((b.y1 - offset[1]) * scale_y, 0)
        b.x2 = min((b.x2 - offset[0]) * scale_x, img_w)
        b.y2 = min((b.y2 - offset[1]) * scale_y, img_h)

    return boxes

#==============Fuctions SUPPORT POSE ESTIMATE ==========
#@Timer("rtmpose preprocesss batch")
def preprocess_pose_batch_fast(images, type="normal"):
    batch = None
    batch_size = len(images)
    if type == "mini":
        batch = np.zeros((batch_size, input_pose_h_mini, input_pose_w_mini, 3), dtype=np.float32)
    else:
        batch = np.zeros((batch_size, input_pose_h, input_pose_w, 3), dtype=np.float32)

    offsets = np.zeros((batch_size, 2), dtype=np.int32)
    
    mean_arr = np.array(mean_pose, dtype=np.float32).reshape(1, 1, 1, 3)
    std_arr = np.array(std_pose, dtype=np.float32).reshape(1, 1, 1, 3)
    
    for i, img in enumerate(images):
        resized_img = None 
        if type == "mini": 
            resized_img, offset_x, offset_y = resize_fast(img, input_pose_w_mini, input_pose_h_mini)
        else:
            resized_img, offset_x, offset_y = resize_fast(img, input_pose_w, input_pose_h)
            
        batch[i] = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        offsets[i] = [offset_x, offset_y]
    
    batch = ((batch - mean_arr) / std_arr).transpose(0, 3, 1, 2)
    
    return batch, offsets


# @Timer("rtmpose postprocess")
def postprocess_pose_batch_fast(simcc_x, simcc_y, img_shapes, offsets, type="normal"):
    batch_size, num_points, x_dim = simcc_x.shape
    _, _, y_dim = simcc_y.shape
    # Argmax
    max_x_pos = np.argmax(simcc_x, axis=2)  # shape: (B, num_points)
    max_y_pos = np.argmax(simcc_y, axis=2)  # shape: (B, num_points)

    # Pose coords
    pose_x = max_x_pos // 2
    pose_y = max_y_pos // 2

    # Score
    score_x = np.take_along_axis(simcc_x, max_x_pos[:, :, None], axis=2).squeeze(2)
    score_y = np.take_along_axis(simcc_y, max_y_pos[:, :, None], axis=2).squeeze(2)
    scores = np.maximum(score_x, score_y)  # shape: (B, num_points)

    # Output
    batch_results = []

    for b in range(batch_size):
        img_h, img_w = img_shapes[b]
        offset_x, offset_y = offsets[b]
        px = pose_x[b]
        py = pose_y[b]
        sc = scores[b]
        x_scale, y_scale = 0, 0
        if type == "mini":
            x_scale = img_w / (input_pose_w_mini - 2 * offset_x)
            y_scale = img_h / (input_pose_h_mini - 2 * offset_y)
        else:
            x_scale = img_w / (input_pose_w - 2 * offset_x)
            y_scale = img_h / (input_pose_h - 2 * offset_y)

        points = [
            {
                'x': float((px[i] - offset_x) * x_scale),
                'y': float((py[i] - offset_y) * y_scale),
                'score': float(sc[i])
            }
            for i in range(num_points)
        ]
        batch_results.append(points)

    return batch_results


def draw_keypoints(image, pose_result):
    for point in pose_result:
        if point['score'] > 0.1:
            cv2.circle(image, (int(point['x']), int(point['y'])), 5, (0, 255, 0), -1)
    return image


#===================Behind stage DET-POSE ==================
# @Timer("rtmdet crop boxes")
def crop_boxes_fast(frame,
                    boxes,
                    score_thresh=0.4,
                    cls_filter={0},
                    max_people=15,
                    min_width=50,
                    min_height=50,
                    return_bboxes=False,
                    scale=1.2):  
    if not boxes:
        return [], [] if return_bboxes else []

    H, W = frame.shape[:2]
    data = np.array(
        [(b.x1, b.y1, b.x2, b.y2, b.cls, b.conf) for b in boxes],
        dtype=np.float32
    )
    x1, y1, x2, y2, cls, conf = data.T

    widths = x2 - x1
    heights = y2 - y1

    valid = (conf >= score_thresh) \
            & (widths >= min_width) & (heights >= min_height) \
            & (x1 < x2) & (y1 < y2) \
            & (x1 < W) & (y1 < H) \
            & (x2 > 0) & (y2 > 0)

    if cls_filter is not None:
        valid &= np.isin(cls.astype(np.int32), list(cls_filter))

    idxs = np.nonzero(valid)[0]
    if idxs.size == 0:
        return [], [] if return_bboxes else []

    areas = widths[idxs] * heights[idxs]
    top_idxs = idxs[np.argsort(-areas)[:max_people]]

    x1 = x1[top_idxs]
    y1 = y1[top_idxs]
    x2 = x2[top_idxs]
    y2 = y2[top_idxs]
    conf = conf[top_idxs]

    # Scale bounding boxes
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w_half = (x2 - x1) / 2 * scale
    h_half = (y2 - y1) / 2 * scale

    x1 = np.clip(cx - w_half, 0, W - 1).astype(np.int32)
    y1 = np.clip(cy - h_half, 0, H - 1).astype(np.int32)
    x2 = np.clip(cx + w_half, 1, W).astype(np.int32)
    y2 = np.clip(cy + h_half, 1, H).astype(np.int32)
    conf = conf.clip(0, 1).astype(np.float32)

    crops = []
    bboxes = []
    for xi1, yi1, xi2, yi2, confi in zip(x1, y1, x2, y2, conf):
        crop = frame[yi1:yi2, xi1:xi2]
        crops.append(crop)
        bboxes.append((xi1, yi1, xi2, yi2, confi))

    return (crops, bboxes) if return_bboxes else crops
