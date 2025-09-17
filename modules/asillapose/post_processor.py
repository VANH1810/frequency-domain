import torch
import math
import numpy as np
import cv2
import random
from modules.common.constants import *

def draw_keypoints(image, skeleton=COCO_SKELETON, colors=VIS_COLORS, bboxes=None, keypoints=None, min_conf=0.5):
    box_thinkness= max(round(0.001111 * min(image.shape[0], image.shape[1])), 1)
    kps_thinkness= max(round(0.002222 * min(image.shape[0], image.shape[1])), 2)
    radius_circle = min(kps_thinkness * 2, kps_thinkness + 3)
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
        for i in range(len(keypoints)):
            keypoints = np.array(keypoints)
            # Draw keypoints and skeletons
            for ic, pair in enumerate(skeleton):
                kp1 = keypoints[i, pair[0]]
                kp2 = keypoints[i, pair[1]]
                if len(kp1) == 3 and len(kp2) == 3:
                    score1 = kp1[2]
                    score2 = kp2[2]
                    # Draw keypoints only if their scores exceed the threshold
                    if score1 > min_conf:
                        cv2.circle(image, (int(kp1[0]), int(kp1[1])), radius_circle, (0, 127, 0), -1)  # Red color for keypoints
                    if score2 > min_conf:
                        cv2.circle(image, (int(kp2[0]), int(kp2[1])), radius_circle, (0, 127, 0), -1)
                    if score1 > min_conf and score2 > min_conf:
                        cv2.line(image, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), colors[ic], kps_thinkness)  # Blue color for skeletons
                else:
                    cv2.circle(image, (int(kp1[0]), int(kp1[1])), radius_circle, (0, 127, 0), -1)
                    cv2.circle(image, (int(kp2[0]), int(kp2[1])), radius_circle, (0, 127, 0), -1)
                    cv2.line(image, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), colors[ic], kps_thinkness)  # Blue color for skeletons

    return image

def postprocess(
        predictions:torch.Tensor,
        conf_thres=0.3,
        oks_thres=0.75,
        nc=1,
        max_det = 1280,
        kpts_shape = (17,3),
        kpt_sigmas=None
    ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Args:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels.

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """
        # Checks
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= oks_thres <= 1, f"Invalid IoU {oks_thres}, valid values are between 0.0 and 1.0"
        # sort predictions by confident score
        scores = predictions[:, :, 0]
        _, idxs  = scores.sort(dim=1, descending=True)
        new_idxs = idxs.unsqueeze(-1).expand(predictions.shape)
        predictions = torch.gather(predictions, dim=1, index=new_idxs)
        bs = predictions.size(0)  # batch size (BCN, i.e. 1,84,6300)
        outputs = []
        for i in range(bs):
            pred = predictions[i]
            scores = pred[:, :nc]
            kpts = pred[:, nc:]
            idxs, _ = torch.where(scores > conf_thres)
            num_det = min(idxs.max(), max_det) if idxs.numel() > 0 else 0
            scores = scores[:num_det]
            kpts = kpts[:num_det].view(num_det, kpts_shape[0], 3)
            clss = torch.zeros((num_det, 1), device=pred.device)
            bboxes = get_pose_bboxes(kpts)
            areas = abs(bboxes[..., 2] - bboxes[..., 0]) * abs(bboxes[..., 3] - bboxes[..., 1])
            keep = oks_nms(kpts, scores, areas, kpt_sigmas, oks_thres)
            # print("KEEP: ", keep)
            if keep.numel() == 0:
                outputs.append(torch.empty((0, 4 + 1 + 1 + kpts.shape[1]), device=pred.device))
                continue
            bboxes, scores, clss, kpts = bboxes[keep], scores[keep], clss[keep], kpts.view(num_det, kpts_shape[0]*kpts_shape[1])[keep]
            # bboxes, scores, cls, kps = preds[:, :4], preds[:, 4], preds[:, 5], preds[:, 6:].reshape(-1, *self.args.kpts_shape)
            output = torch.cat([bboxes, scores, clss, kpts], dim=1)
            outputs.append(output)
        return outputs

COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

def compute_oks(kpts1, kpts2, area, kpt_sigmas, conf=0.5):
    kpt_mask:torch.Tensor = kpts1[..., 2] >= conf
    """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
    d = (kpts1[..., 0] - kpts2[..., 0]).pow(2) + (kpts1[..., 1] - kpts2[..., 1]).pow(2)
    e = d / (2 * kpt_sigmas).pow(2) / ((area + 1e-9) * 2)  # from cocoeval
    oks = torch.exp(-e) * kpt_mask
    return oks.sum(dim=-1) / (kpt_mask.sum(dim=-1) + 1e-9)


def oks_nms(kpts_list, scores, areas, kpt_sigmas, oks_threshold=0.75):
    if kpt_sigmas is None:
        kpt_sigmas = torch.tensor(COCO_SIGMAS, device=kpts_list.device)
    else:
        kpt_sigmas = kpt_sigmas.to(kpts_list.device)
    keep = []
    indices = torch.arange(0, scores.size(0), 1, device=kpts_list.device)
    while indices.numel() > 0:
        idx = indices[0]
        keep.append(idx.item())
        if indices.numel() == 1:
            break
        # Batch compute OKS for remaining keypoints
        remaining_kpts = kpts_list[indices[1:]]
        areas_sum = (areas[idx] + areas[indices[1:]]).view(-1, 1)
        oks_values = compute_oks(
            kpts_list[idx].unsqueeze(0),  # Shape (1, N, 3)
            remaining_kpts,               # Shape (M, N, 3)
            areas_sum,
            kpt_sigmas
        )

        # Keep only indices with OKS < threshold
        indices = indices[1:][oks_values < oks_threshold]
    return torch.tensor(keep)

def get_pose_bboxes(kpts, conf_threshold=0.5, expand_ratio=0.3):
    # Ensure 4D shape (Batch-size, N, 17, 3)
    is_unsqueeze = False
    if kpts.dim() == 3:
        is_unsqueeze = True
        kpts = kpts.unsqueeze(0)  # Add batch dimension if missing
    # Extract (x, y) and confidence scores
    xy = kpts[:, :, :, :2]  # Shape (N, 17, 2)
    conf = kpts[:, :, :, 2]  # Shape (N, 17)
    # Mask: Keep keypoints where conf > threshold, else set to large/small values
    mask = conf > conf_threshold  # Shape (N, 17)

    # Replace invalid keypoints with extreme values
    xy_masked = xy.clone()
    xy_masked[~mask] = 1e9  # Large value for min
    x_min = xy_masked[:, :, :, 0].min(dim=2).values
    y_min = xy_masked[:, :, :, 1].min(dim=2).values

    xy_masked[~mask] = -1e9  # Small value for max
    x_max = xy_masked[:, :, :, 0].max(dim=2).values
    y_max = xy_masked[:, :, :, 1].max(dim=2).values

    # Compute bbox width and height
    width = x_max - x_min
    height = y_max - y_min

    # Expand by 25%
    x_min = x_min - (width * expand_ratio / 2)
    y_min = y_min - (height * expand_ratio / 2)
    x_max = x_max + (width * expand_ratio / 2)
    y_max = y_max + (height * expand_ratio / 2)

    # Stack into (N, 4)
    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    if is_unsqueeze:
        bboxes = bboxes.squeeze(0)
    return bboxes
    
    
def compute_iou(box1, box2, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)

def resize_letter_box(image:np.ndarray, 
                      output_shape, 
                      padding_color=PADDING_VAL, 
                      padding_type="center", 
                      interpolation=cv2.INTER_LINEAR):
    
    if isinstance(padding_color, int):
        padding_color = (padding_color, padding_color, padding_color)
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)
    img_height, img_width = image.shape[:2]
    target_height, target_width = output_shape
    
    scale = min(target_width / img_width, target_height/ img_height)
    # Resize the image to the new dimensions
    new_width = round(scale * img_width)
    new_height = round(scale * img_height)
    if interpolation is None:
        interpolation = random.choice([cv2.INTER_LINEAR, 
                                       cv2.INTER_NEAREST,
                                       cv2.INTER_CUBIC,
                                       cv2.INTER_AREA, 
                                       cv2.INTER_LANCZOS4])
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    if padding_type == "random":
        pad_top = random.randint(0, target_height - new_height)
        pad_left = random.randint(0, target_width - new_width)
    elif padding_type == "center":
        pad_top = (target_height - new_height) // 2
        pad_left = (target_width - new_width) // 2
    else:
        pad_top, pad_left = 0, 0
    pad_bottom = target_height - new_height - pad_top
    pad_right = target_width - new_width - pad_left
    # Apply padding
    padded_image = cv2.copyMakeBorder(
        resized_image,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color
    )
    padding = [pad_top, pad_bottom, pad_left, pad_right]
    return padded_image, scale, padding

def transform_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))  # Change image layout to (channels, height, width)
    image = image.astype(np.float32) / 255.0  # Normalize to range [0, 1]
    image = np.ascontiguousarray(image)  # Ensure contiguous memory layout
    return image


def get_result(results, scale, top, left, num_kpts=17):
    if results is None or len(results) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]) 
    
    output = np.array(results)

    if output.ndim < 2 or output.shape[1] < (6 + num_kpts * 3):
        raise ValueError(f"Invalid results shape: {output.shape}, expected at least {(6 + num_kpts * 3)} columns")

  
    row_sums = np.sum(output, axis=1)
    output = output[row_sums != 0.0]

    if output.shape[0] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])  

    bboxes = output[:, 0:4].copy()  
    cls = output[:, 5].astype(int)  
    scores = output[:, 4]
    kps = output[:, 6:].reshape(-1, num_kpts, 3)  


    bboxes[..., 0::2] -= left  
    bboxes[..., 1::2] -= top 
    kps[..., 0] -= left
    kps[..., 1] -= top

    bboxes /= scale
    kps[..., :2] /= scale  

    # Làm tròn số nguyên
    bboxes = np.round(bboxes).astype(int)
    kps[..., :2] = np.round(kps[..., :2]).astype(int)
    kps[..., 2] = np.round(kps[..., 2], 2)  

    
    if scores.shape[0] > 2:
        scores[..., 0] = np.round(scores[..., 2], 0)

    # np.set_printoptions(suppress=True, precision=2)  # Định dạng in ra đẹp hơn
    return bboxes, scores, cls, kps