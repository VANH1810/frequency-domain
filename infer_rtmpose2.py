import cv2
import json
#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import torch
from modules.rtmpose.rtmdet_infer import RTMDetClient
from modules.rtmpose.rtmpose_infer import RTMPoseClient 
from modules.rtmpose.rtm_utils import crop_boxes_fast, preprocess_pose_batch_fast
from DuyTan4.Yolov7_tracker.tracker.trackers.byte_tracker import ByteTracker
from helper import _draw_limbs
import argparse
import numpy as np
import csv
import os
np.float = float

class _Box:
    def __init__(self, x1, y1, x2, y2, cls, conf):
        self.x1 = x1; self.y1 = y1
        self.x2 = x2; self.y2 = y2
        self.cls = cls; self.conf = conf
def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--tracker', type=str, default='sort', help='sort, deepsort, etc')
    parser.add_argument('--reid', type=bool, default=True, help='enable reid model, work in bot, byte, ocsort and hybridsort')
    parser.add_argument('--reid_model', type=str, default='osnet_x0_25', help='osnet or deppsort')

    parser.add_argument('--kalman_format', type=str, default='byte', help='use what kind of Kalman, sort, deepsort, byte, etc.')
    parser.add_argument('--img_size', type=int, default=1280, help='image size, [h, w]')

    parser.add_argument('--conf_thresh', type=float, default=0.2, help='filter tracks')
    parser.add_argument('--conf_thresh_low', type=float, default=0.1, help='filter low conf detections, used in two-stage association')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU thresh to filter tracks')

    parser.add_argument('--device', type=str, default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')


    """model path"""
    # other model path
    parser.add_argument('--reid_model_path', type=str, default='DuyTan4/Yolov7_tracker/weights/osnet_x0_25.pth', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth', help='path of DHN path for DeepMOT')

   
    """other options"""
    parser.add_argument('--fuse_detection_score', type=bool, default=False, help='fuse detection conf with iou score')
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')

    parser.add_argument('--save_dir', type=str, default='track_demo_results')
    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')
    
    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')

    """camera parameter"""
    parser.add_argument('--camera_parameter_folder', type=str, default='./modules/tracker/cam_param_files', help='folder path of camera parameter files')
    return parser.parse_args()

#coco = COCO(coco_gt_path)
#img_ids = coco.getImgIds()

def convert_to_coco_format(image_id, track_ids, boxes, scores, keypoints):
    """Convert tracking+pose results to COCO-format dicts."""
    results = []
    for tid, box, score, kps in zip(track_ids, boxes, scores, keypoints):
        x1,y1,x2,y2 = [float(x) for x in box]
        w, h = x2-x1, y2-y1
        flat_kps = []
        for kp in kps:
            flat_kps.extend([int(kp[0]), int(kp[1]), kp[2]])
        results.append({
            "image_id": int(image_id),
            "category_id": 1,
            "bbox": [x1, y1, w, h],
            "score": float(score),
            "keypoints": flat_kps,
            "track_id": int(tid)
        })
    return results

def write_csv_data(csv_path, info):
    """Append tracking & pose info to CSV."""
    if not os.path.exists(csv_path):
        with open(csv_path, 'a') as write:
            writer = csv.writer(write)
            writer.writerow(['FrameID','PID','Bbox','Pose'])
    else:
        with open(csv_path, 'a') as write:
            writer = csv.writer(write)
            for data in info:
                frmid = data['image_id']
                bbox = data['bbox']
                score = data['score']
                bbox = list(bbox)
                bbox.append(score)
                pid = data['track_id']
                kps = data['keypoints']
                writer.writerow([frmid,pid,bbox, kps])


if __name__ == '__main__':
    args = get_args()
    
    # 1) Initialize Triton clients once
    camera_name = "cam2"
    det_client = RTMDetClient(url="localhost:8001", camera_name=camera_name)
    pose_client = RTMPoseClient(url="localhost:8001",
                            model="rtmpose_mini", model_version="1",
                            model_type="mini", use_threads=True, max_batch_size=5, num_infer_threads=4)

    # 2) Gather all videos in folder
    video_folders = [
       'recorded_data'
    ]

    video_paths = [
    ]
    for folder in video_folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file == 'color.avi':
                    video_paths.append(os.path.join(root, file))

    print(video_paths)
    output_folder = 'infer-robot-dog'
    os.makedirs(output_folder, exist_ok=True)

    for vid_path in video_paths:
        parent_folder = os.path.basename(os.path.dirname(vid_path))  # tên folder chứa color.avi
        vid_file = os.path.basename(vid_path)                        # "color.avi"
        vid_name = f"{parent_folder}_{vid_file.split('.avi')[0]}"    # ghép tên folder + "color"
        
        csv_file = f'{vid_name}.csv'
        csv_path = os.path.join(output_folder, csv_file)
        
        print("Input video:", vid_path)
        print("Output CSV:", csv_path)

        # out_video = os.path.join(save_folder, f"{vid_name}_annotated.mp4")

        # Open video
        vid = cv2.VideoCapture(vid_path)
        if not vid.isOpened():
            print(f"[Error] Cannot open {vid_path}")
            continue
        
        # Get FPS (fallback to 5.0 if zero or invalid)
        fps = vid.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 5.0   

        # Setup tracker for this video
        tracker = ByteTracker(args, frame_rate=fps)

        # Prepare VideoWriter
        # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

        print(vid_path)
        frame_index = 1
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            h_frame, w_frame = frame.shape[:2]
            # 1) Detection
            _, boxes = det_client.detect(frame, frame_index=frame_index, conf_thresh=0.4, iou_thresh=0.4)
            # print(f"[DEBUG] Detected {len(boxes)} boxes")
            # for i, box in enumerate(boxes):
            #     print(f"  Box{i}: ({box.x1:.1f},{box.y1:.1f})-({box.x2:.1f},{box.y2:.1f}), conf={box.conf:.2f}")

            frame_index += 1
            
            if len(boxes) == 0:
                continue

            # 2) Prepare detections for tracker
            boxes_org, scores_org, kps_org = [], [], []
            detections = []
            for box in boxes:
                x1, y1 = box.x1, box.y1
                x2, y2 = box.x2, box.y2
                score = box.conf     

                # [x, y, w, h, score]

                # Clamp về [0, w_frame-1] / [0, h_frame-1]
                x1c = int(max(0, min(x1, w_frame-1)))
                y1c = int(max(0, min(y1, h_frame-1)))
                x2c = int(max(0, min(x2, w_frame-1)))
                y2c = int(max(0, min(y2, h_frame-1)))

                # Bỏ qua nếu sau clamp thành invalid box
                if x2c <= x1c or y2c <= y1c:
                    print(f"[WARN-DET] invalid det bbox after clamp: ({x1c},{y1c})-({x2c},{y2c})")
                    continue

                # Tính lại w,h từ coords clamp
                wc = x2c - x1c
                hc = y2c - y1c

                detections.append([x1c, y1c, wc, hc, score, 0])
                
            detections = np.array(detections)
            online_targets = tracker.update(detections,frame, frame)

            tracked_bboxes = []
            tracked_ids = []
            box_objs = []
            crops = []
            # print(f"[DEBUG] Tracker returned {len(online_targets)} targets")
            for t in online_targets:
                tlwh = t.tlwh
                tid  = t.track_id
                if tlwh[2]*tlwh[3] < args.min_area: 
                    continue
                x1, y1, w, h = tlwh
                x1i = int(max(0, min(x1, w_frame-1)))
                y1i = int(max(0, min(y1, h_frame-1)))
                x2i = int(max(0, min(x1 + w, w_frame-1)))
                y2i = int(max(0, min(y1 + h, h_frame-1)))

                if x2i > x1i and y2i > y1i:
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        crops.append(crop)
                        tracked_bboxes.append([x1i, y1i, x2i, y2i, t.score])
                        tracked_ids.append(tid)
                        box_objs.append(_Box(x1i, y1i, x2i, y2i, cls=0, conf=t.score))
                    else:
                        print(f"[WARN] Empty crop after clamp: {crop.shape} from ({x1i},{y1i},{x2i},{y2i})")
                else:
                    print(f"[WARN] Invalid bbox after clamp: ({x1i},{y1i},{x2i},{y2i})")

                # print(f"[DEBUG] Frame {frame_index}, Track {tid}: "
                #     f"bbox_clamped=({x1i},{y1i})-({x2i},{y2i}), frame_size=({w_frame},{h_frame})")
            # print("number tracked_bboxes: ", len(tracked_bboxes))
            # print("number tracked_crop: ", len(crops))
            # print(detections)
            # box_objs = []
            # for (x1, y1, x2, y2, score) in tracked_bboxes:

            
            # crops, crop_bboxes = crop_boxes_fast(
            #     frame,
            #     box_objs,
            #     score_thresh=0.2,
            #     cls_filter={0},
            #     max_people=10,
            #     min_width=10,
            #     min_height=10,
            #     scale=1.0,
            #     return_bboxes=True
            # )

            if len(crops) == 0:
                continue
            inp, offsets = preprocess_pose_batch_fast(crops, "mini")
            is_preprocessed = True

            _, keypoints = pose_client.estimate_pose(
                crops, inp, offsets, is_preprocessed, frame_index=frame_index)
            track_keypoints = []

            for box, kp in zip(tracked_bboxes,keypoints):
                x1,y1,x2,y2,score = box
                w = x2 - x1
                h = y2 - y1
                # detections.append([x1, y1, w, h, score,0])
                kps = [[p['x'] +x1 , p['y'] + y1, round(p["score"],2)] for p in kp]
                track_keypoints.append(kps)
                # frame = _draw_limbs(kps, frame)


            for (x1, y1, x2, y2, conf), kps, tid in zip(tracked_bboxes, track_keypoints, tracked_ids):
                w = x2 - x1
                h = y2 - y1

                boxes_org.append([x1, y1, x1+w, y1+h])
                scores_org.append(conf)
                # kps_org.append([[int(p["x"] + x1), int(p["y"] + y1), round(p["score"], 2)] for p in kp])
                kps_org.append(kps)
                # kps = [[p['x'] , p['y'], p] for p in kp]

                # Vẽ skeleton cho từng người
                #frame = _draw_limbs(kps, frame)

                # Vẽ bounding box
                #frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Vẽ ID ngay trên bbox
                # frame = cv2.putText(
                #     frame, f"ID{tid}",
                #     (int(x1), int(y1) - 5),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                # )

            results = convert_to_coco_format(
                image_id=frame_index,
                track_ids=tracked_ids,
                boxes=boxes_org,
                scores=scores_org,
                keypoints=kps_org
            )

            write_csv_data(csv_path, results)
            # frame_index += 1
            # writer.write(frame)
        
        vid.release()
        # writer.release()