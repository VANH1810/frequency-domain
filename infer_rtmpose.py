import cv2
import json
#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

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
    results = []
    for tid, box, score, kps in zip(track_ids, boxes, scores, keypoints):
        x1, y1, x2, y2 = [float(x) for x in box]
        w, h = x2 - x1, y2 - y1
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
    camera_name = "cam1"
    det_client = RTMDetClient(url="localhost:8001", camera_name=camera_name)
    pose_client = RTMPoseClient(url="localhost:8001",
                            model="rtmpose_mini", model_version="1",
                            model_type="mini", use_threads=True, max_batch_size=5, num_infer_threads=4)
    args = get_args()
    # writer = cv2.VideoWriter('test.mp4', writer_fourcc, fps, (width, height))
    video_folder = '/workspace/data/client_v4_video/5FPS_all'
    video_paths = ['data/client_v3_video/Staggering_Heavy_Back_Walking_2.MP4']
    # for root, folders, files in os.walk(video_folder):
    #     for file in files:
    #         if file.endswith('.MP4'):
    #             video_paths.append(os.path.join(root, file))
    
    for vid_path in video_paths:

        vid_file = vid_path.split('/')[-1]
        vid_name = vid_file.split('.mp4')[0]
        csv_file = f'{vid_name}.csv'
        save_folder = vid_path[:vid_path.index(vid_file) -1]
        save_folder = save_folder.replace(video_folder, f'{video_folder}_csv')
        os.makedirs(save_folder, exist_ok=True)
        csv_path = os.path.join(save_folder, csv_file)
        vid = cv2.VideoCapture(vid_path)

        fps = vid.get(cv2.CAP_PROP_FPS)
        tracker = ByteTracker(args,frame_rate=fps)
        writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter('test3.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (width, height))
        if not vid.isOpened():
            print("Error: Could not open video.")
            exit()
        frame_index = 1
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            _, boxes = det_client.detect(frame, frame_index=frame_index, conf_thresh=0.25, iou_thresh=0.35)
            if len(boxes) == 0:
                continue
            boxes_org, scores_org, kps_org = [], [], []
            # detections = []
            box_objs = []
            for box in boxes:
                x1, y1 = box.x1, box.y1
                x2, y2 = box.x2, box.y2
                score = box.conf     
                # 2) Chuyển về [x, y, w, h, score]
                w, h = x2 - x1, y2 - y1
                # detections.append([x1, y1, w, h, score])
                box_objs.append(_Box(x1, y1, x2,y2, cls=0, conf=score))
                
            # detections = np.array(detections)
            # print(vid_name)
            # print(detections)
            # box_objs = []
            # for (x1, y1, x2, y2, score) in tracked_bboxes:
            #     box_objs.append(_Box(x1, y1, x2, y2, cls=0, conf=score))

            
            crops, crop_bboxes = crop_boxes_fast(
                frame,
                box_objs,
                score_thresh=0.2,
                cls_filter={0},
                max_people=10,
                min_width=10,
                min_height=10,
                return_bboxes=True
                )
            if len(crops) == 0:
                continue
            inp, offsets = preprocess_pose_batch_fast(crops, "mini")
            is_preprocessed = True

            _, keypoints = pose_client.estimate_pose(
                crops, inp, offsets, is_preprocessed, frame_index=frame_index
            )
            detections = []
            for box, kp in zip(crop_bboxes,keypoints):
                x1,y1,x2,y2,score = box
                w = x2 - x1
                h = y2 - y1
                detections.append([x1, y1, w, h, score])
                # kps = [[p['x'] +x1 , p['y'] + y1] for p in kp]
                # frame = _draw_limbs(kps, frame)
            detections = np.array(detections)
            online_targets = tracker.update(detections,frame, frame)
            # lấy bbox đã track
            tracked_bboxes = []
            tracked_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid  = t.track_id
                # filter area nhỏ
                if tlwh[2]*tlwh[3] < args.min_area: continue
                x1, y1, w, h = tlwh
                tracked_bboxes.append([x1, y1, x1+w, y1+h, t.score])
                tracked_ids.append(tid)
                frame = cv2.putText(frame, f"ID{tid}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            for (x1, y1, x2, y2, conf), kp, tid in zip(crop_bboxes, keypoints, tracked_ids):
                boxes_org.append([x1, y1, x2, y2])
                scores_org.append(conf)
                kps_org.append([[int(p["x"] + x1), int(p["y"] + y1), round(p["score"], 2)] for p in kp])
                kps = [[p['x'] +x1 , p['y'] + y1] for p in kp]
                frame = _draw_limbs(kps, frame)
                frame = cv2.putText(frame, f"ID{tid}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                frame = cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0),2)
            results = convert_to_coco_format(image_id=frame_index,
                    track_ids=tracked_ids,
                    boxes=boxes_org,
                    scores=scores_org,
                    keypoints=kps_org)
            # write_csv_data(csv_path, results)
            frame_index +=1
            writer.write(frame)
        
        vid.release()

