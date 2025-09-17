from helper import _draw_limbs
import ast
import cv2
import os
import re
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FONT_PATH = os.path.join(_BASE_DIR, "source", "fonts", "epkgobld.ttf")
ALLOWED_VID_EXTS = ('.mp4','.mov','.avi','.mkv')

LINE_COLOR = [
    (0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
    (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77),
    (77, 255, 77), (77, 222, 255), (255, 156, 127), (0, 127, 255),
    (255, 127, 77), (0, 77, 255), (127, 127, 255), (255, 0, 127),
    (0, 127, 0), (255, 255, 128), (0, 0, 50), (0, 150, 50)
]

def get_font(size=26):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        # fallback nếu thiếu file, tránh crash
        return ImageFont.load_default()

# font mặc định dùng cho _draw_action / _draw_label
font = get_font(15)

def find_video_file(video_dir: str, vid_base: str):
    """Tìm file video theo tên base (đã bỏ .mp4), trả về path hoặc None."""
    for ext in ALLOWED_VID_EXTS:
        p = os.path.join(video_dir, vid_base + ext)
        if os.path.exists(p):
            return p
    # fallback: quét thư mục để phòng tên khác hoa/thường
    for fn in os.listdir(video_dir):
        b, e = os.path.splitext(fn)
        if e.lower() in ALLOWED_VID_EXTS and canon_vid(b) == canon_vid(vid_base):
            return os.path.join(video_dir, fn)
    return None

def _parse_frame_range(fr_str: str):
    s = str(fr_str).strip()
    a, b = s.split('-')
    return int(a), int(b)


def canon_vid(name: str):
    s = str(name).strip()
    s = os.path.basename(s)                         # bỏ path nếu có
    s = re.sub(r'\.(mp4|mov|avi|mkv)$', '', s, flags=re.IGNORECASE)  # bỏ đuôi video
    return s

def resolve_results_csv(results_dir_or_file: str, vid_base: str):
    # Nếu là file CSV thì trả về luôn
    if os.path.isfile(results_dir_or_file):
        return results_dir_or_file
    # Nếu là thư mục: ghép chuẩn tên <vid_base>_eval.csv
    base = canon_vid(vid_base)
    return os.path.join(results_dir_or_file, f"{base}_eval.csv")

def _draw_pid(img, box, identity=None, thickness=1, offset=(0, 0),
              txt_color=(255,255,255), bg_color=(0,0,0), alpha=0.55,
              anchor="top_out", size=26, pad=4):
    # box: [x1,y1,x2,y2]
    x1, y1, x2, y2 = [int(i + (offset[idx % 2])) for idx, i in enumerate(box)]
    if identity is None:
        return img

    H, W = img.shape[:2]
    label = str(identity)

    pil  = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    fnt  = get_font(size)

    # đo kích thước text
    _, _, tw, th = draw.textbbox((0, 0), label, font=fnt)

    # ------- chọn vị trí ngoài bbox -------
    def pos_top():
        tx = int(x1 + (x2 - x1 - tw) / 2)
        ty = y1 - th - 2*pad
        return tx, ty

    def pos_bottom():
        tx = int(x1 + (x2 - x1 - tw) / 2)
        ty = y2 + 2
        return tx, ty

    def pos_left():
        tx = x1 - tw - 2*pad
        ty = max(0, min(H - th - 2*pad, y1))
        return tx, ty

    def pos_right():
        tx = x2 + 2
        ty = max(0, min(H - th - 2*pad, y1))
        return tx, ty

    # chọn theo anchor, có fallback nếu tràn khung
    if anchor in ("top_out", "top"):
        tx, ty = pos_top()
        if ty < 0: anchor = "bottom_out"
    if anchor == "bottom_out":
        tx, ty = pos_bottom()
        if ty + th + 2*pad > H: anchor = "right_out"
    if anchor == "left_out":
        tx, ty = pos_left()
        if tx < 0: anchor = "right_out"
    if anchor == "right_out":
        tx, ty = pos_right()
        if tx + tw + 2*pad > W:  # last resort: cố nhét trên trong phạm vi
            tx, ty = pos_top()

    # clamp cuối để bảo đảm nằm trong khung
    tx = max(0, min(W - tw - 2*pad, tx))
    ty = max(0, min(H - th - 2*pad, ty))

    # nền mờ
    overlay = pil.copy()
    rx1, ry1 = tx - pad, ty - pad
    rx2, ry2 = tx + tw + pad, ty + th + pad
    ImageDraw.Draw(overlay).rounded_rectangle((rx1, ry1, rx2, ry2), radius=6, fill=bg_color)
    pil = Image.blend(pil, overlay, alpha)

    # vẽ chữ
    draw = ImageDraw.Draw(pil)
    draw.text((tx, ty), label, font=fnt, fill=txt_color)

    img[:,:,:] = np.array(pil)
    return img

def build_pred_frame_map(results_csv_path: str):
    """
    Đọc file kết quả {vid}_eval.csv (đã remap sang frame thực).
    Trả về: pred_map[(pid, frame)] = {'label': 'Run'/'Walk', 'score': float}
    Rule: nếu có >=1 cửa sổ 'Run' phủ frame → chọn Run với score_min (càng âm càng “abnormal”).
          nếu không có Run → Walk (lấy score lớn nhất nếu muốn).
    """
    if not os.path.exists(results_csv_path):
        return {}
    df = pd.read_csv(results_csv_path)
    if 'pid' not in df.columns or 'frame_range' not in df.columns or 'Pred' not in df.columns:
        return {}
    df['pid'] = df['pid'].astype(str).str.strip()

    pred_map = {}  # (pid, frame) -> {'label','score'}
    for _, r in df.iterrows():
        st, ed = _parse_frame_range(r['frame_range'])
        pid = str(r['pid'])
        pred = str(r['Pred'])
        sc = float(r.get('score', 0.0))
        for f in range(int(st), int(ed)+1):
            key = (pid, f)
            cur = pred_map.get(key)
            if pred == 'Run':
                # ưu tiên Run + score nhỏ hơn (âm hơn)
                if (cur is None) or (cur['label'] != 'Run') or (sc < cur['score']):
                    pred_map[key] = {'label': 'Run', 'score': sc}
            else:
                # Walk chỉ gán nếu chưa có gì
                if cur is None:
                    pred_map[key] = {'label': 'Walk', 'score': sc}
    return pred_map

def build_gold_frame_map(gold_csv_path: str, vid_base: str):
    """
    Gold có các interval Run; mặc định ngoài khoảng là Walk.
    Trả về: gold_map[(pid, frame)] = 'Run'/'Walk'
    """
    gmap = {}
    if (gold_csv_path is None) or (not os.path.exists(gold_csv_path)):
        return gmap  # rỗng nghĩa là coi như unknown → hiển thị '—'
    gdf = pd.read_csv(gold_csv_path).rename(columns={
        'VideoName':'vid','PID':'pid','Action':'action',
        'Start_frame':'g_st','End_frame':'g_ed'
    })
    gdf['vid'] = gdf['vid'].apply(canon_vid)
    gdf = gdf[gdf['vid'] == canon_vid(vid_base)]
    if gdf.empty:
        return gmap
    gdf['pid'] = gdf['pid'].astype(str).str.strip()
    gdf['g_st'] = pd.to_numeric(gdf['g_st'], errors='coerce').fillna(0).astype(int)
    gdf['g_ed'] = pd.to_numeric(gdf['g_ed'], errors='coerce').fillna(0).astype(int)

    for _, r in gdf.iterrows():
        pid = r['pid']; st=int(r['g_st']); ed=int(r['g_ed'])
        for f in range(st, ed+1):
            gmap[(pid, f)] = 'Run'
    # phần còn lại implicit = Walk (khi lookup không thấy)
    return gmap

# ====== 14-keypoint skeleton (Head, Neck, R/L Shoulders, Elbows, Wrists, Hips, Knees, Ankles) ======
# index map:
# 0:HeadTop, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist, 5:LShoulder, 6:LElbow, 7:LWrist,
# 8:RHip, 9:RKnee, 10:RAnkle, 11:LHip, 12:LKnee, 13:LAnkle

L_PAIR_14 = [
    [0, 1,  "H",  "HEAD"],
    [1, 2,  "RS", "RIGHT_SHOULDER"],  [2, 3,  "RE", "RIGHT_ELBOW"],  [3, 4,  "RW", "RIGHT_WRIST"],
    [1, 5,  "LS", "LEFT_SHOULDER"],   [5, 6,  "LE", "LEFT_ELBOW"],   [6, 7,  "LW", "LEFT_WRIST"],
    [1, 8,  "RH", "RIGHT_HIP"],       [8, 9,  "RK", "RIGHT_KNEE"],   [9, 10, "RA", "RIGHT_ANKLE"],
    [1, 11, "LH", "LEFT_HIP"],        [11,12, "LK", "LEFT_KNEE"],    [12,13, "LA", "LEFT_ANKLE"],
    [2, 5,  "SH", "SHOULDERS"],       [8, 11, "HP", "HIPS"]
]

def _draw_limbs_14(current_pose, img, thickness=2):
    """
    current_pose: ndarray (14, 2) -> (x, y)
    Vẽ skeleton 14-kp với cùng style như _draw_limbs (17-kp).
    """
    H, W = img.shape[:2]
    for i, pair in enumerate(L_PAIR_14):
        c_a, c_b, tag = pair[0], pair[1], pair[2]
        if c_a >= len(current_pose) or c_b >= len(current_pose):
            continue
        x0 = int(current_pose[c_a][0]); y0 = int(current_pose[c_a][1])
        x1 = int(current_pose[c_b][0]); y1 = int(current_pose[c_b][1])
        # bỏ điểm rỗng (<=0) hoặc nằm ngoài khung
        if (x0 <= 0 or y0 <= 0 or x1 <= 0 or y1 <= 0 or
            x0 >= W or x1 >= W or y0 >= H or y1 >= H):
            continue

        # hạn chế outlier giãn quá lớn (giữ nguyên logic của bạn)
        if (x1 - x0) > 0.3 * W or (y0 - y1) > 0.3 * H:
            pass

        color = LINE_COLOR[i % len(LINE_COLOR)]
        img = cv2.line(img, (x0, y0), (x1, y1), color, thickness)
        img = cv2.putText(img, tag, (x1, y1), cv2.FONT_HERSHEY_PLAIN,
                          max(0.5, thickness / 5), color, 1)
        img = cv2.circle(img, (x0, y0), 1, color, thickness)
        img = cv2.circle(img, (x1, y1), 1, color, thickness)
    return img

def draw_skeleton_auto(current_pose, img, thickness=2):
    """
    Wrapper: tự chọn hàm vẽ theo số keypoints.
    - >=17 điểm: dùng _draw_limbs (17-kp) sẵn có của bạn
    - ==14 điểm: dùng _draw_limbs_14 (mới)
    - khác: bỏ qua (trả về img)
    """
    n = len(current_pose) if not isinstance(current_pose, np.ndarray) else current_pose.shape[0]
    if n >= 17:
        return _draw_limbs(current_pose, img, thickness)
    elif n == 14:
        return _draw_limbs_14(current_pose, img, thickness)
    else:
        return img

def _draw_info_lines(img, box, lines, size=18,
                     txt_color=(255,255,255), bg_color=(0,0,0),
                     alpha=0.55, pad=4):
    """
    Vẽ block nhiều dòng (pred / true) cạnh bbox.
    box: [x1,y1,x2,y2]
    """
    if not lines:
        return img

    H, W = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # đảm bảo box nằm trong khung ảnh ở mức "vừa phải"
    x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))

    pil  = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    fnt  = get_font(size)

    # đo kích thước block
    boxes = [draw.textbbox((0,0), t, font=fnt) for t in lines]
    line_heights = [(b[3]-b[1]) for b in boxes]
    tw = max((b[2]-b[0]) for b in boxes)
    th = sum(line_heights) + (len(lines)-1)*2  # khoảng cách 2px giữa các dòng

    # ưu tiên vẽ phía trên bbox
    tx = x1
    ty = y1 - th - 2*pad
    if ty < 0:
        ty = min(H - th - 2*pad, y1 + 2)
    tx = max(0, min(W - (tw + 2*pad), tx))

    # nền mờ
    overlay = pil.copy()
    rx1, ry1 = tx - pad, ty - pad
    rx2, ry2 = tx + tw + pad, ty + th + pad
    ImageDraw.Draw(overlay).rounded_rectangle((rx1, ry1, rx2, ry2), radius=6, fill=bg_color)
    pil = Image.blend(pil, overlay, alpha)

    # !!! phải tạo lại draw sau khi blend !!!
    draw = ImageDraw.Draw(pil)

    # vẽ từng dòng
    y = ty
    for i, t in enumerate(lines):
        draw.text((tx, y), t, font=fnt, fill=txt_color)
        y += line_heights[i] + 2

    img[:, :, :] = np.array(pil)
    return img

# ---- Parse POSES -> points & bbox -------------------------------------------
def _poses_str_to_points(poses_str):
    """Chuỗi '[x0,y0,x1,y1,...]' -> ndarray (K,2) float hoặc None."""
    if not isinstance(poses_str, str) or len(poses_str) < 4:
        return None
    try:
        arr = ast.literal_eval(poses_str)
        if not isinstance(arr, (list, tuple)) or len(arr) < 4:
            return None
        arr = [float(v) for v in arr]
        pts = np.array(arr, dtype=float).reshape(-1, 2)
        return pts
    except Exception:
        return None

def _bbox_from_points(pts):
    if pts is None or len(pts) == 0:
        return None
    xs = pts[:,0]; ys = pts[:,1]
    x1, y1 = int(np.floor(xs.min())), int(np.floor(ys.min()))
    x2, y2 = int(np.ceil(xs.max())),  int(np.ceil(ys.max()))
    return [x1, y1, x2, y2]

def load_pose_annots_1based(pose_csv_path: str):
    """
    Trả về:
      frames_map[frame] = list({'pid':str, 'pts':ndarray(K,2) or None, 'bbox':[x1,y1,x2,y2] or None})
    Bảo đảm FRAMEID ra 1-based (nếu file là 0-based sẽ +1).
    """
    dfp = pd.read_csv(pose_csv_path)
    if 'FRAMEID' not in dfp.columns or 'PID' not in dfp.columns:
        raise RuntimeError(f"{pose_csv_path} thiếu FRAMEID/PID")
    dfp['pid'] = dfp['PID'].astype(str).str.strip()
    dfp['FRAMEID'] = pd.to_numeric(dfp['FRAMEID'], errors='coerce').fillna(0).astype(int)

    # suy luận 1-based
    mins = dfp.groupby('pid')['FRAMEID'].min().tolist()
    one_based = (pd.Series(mins).value_counts().idxmax() == 1)
    if not one_based:
        dfp['FRAMEID'] = dfp['FRAMEID'] + 1

    frames_map = {}
    for _, r in dfp.iterrows():
        fr  = int(r['FRAMEID'])
        pid = r['pid']
        pts = _poses_str_to_points(r['POSES']) if 'POSES' in dfp.columns else None
        bb  = _bbox_from_points(pts) if pts is not None else None
        frames_map.setdefault(fr, []).append({'pid': pid, 'pts': pts, 'bbox': bb})
    return frames_map


# ---- Dùng trong visualize_one_video -----------------------------------------
def visualize_one_video(vid_base: str,
                        video_dir: str,
                        pose_csv_path: str,
                        results_dir: str,
                        gold_csv_path: str,
                        out_dir: str,
                        use_draw_pid: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    in_video = find_video_file(video_dir, vid_base)
    if in_video is None:
        print(f"[WARN] Không tìm thấy video cho {vid_base}")
        return

    frames_map = load_pose_annots_1based(pose_csv_path)
    res_csv = resolve_results_csv(results_dir, vid_base)
    pred_map   = build_pred_frame_map(res_csv)
    gold_map   = build_gold_frame_map(gold_csv_path, vid_base)

    print("[DBG] video:", in_video)
    print("[DBG] results csv:", res_csv, "exists=", os.path.exists(res_csv))
    print("[DBG] pose frames:", len(frames_map))
    print("[DBG] pred map entries:", len(pred_map))

    cap = cv2.VideoCapture(in_video)
    if not cap.isOpened():
        print(f"[ERR] Mở video lỗi: {in_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(out_dir, f"{vid_base}_overlay.mp4")
    writer = cv2.VideoWriter(out_path, fourc, float(fps), (W, H))

    font  = cv2.FONT_HERSHEY_SIMPLEX
    red   = (0,0,255)
    green = (0,255,0)
    white = (255,255,255)
    black = (0,0,0)

    f1 = 0
    ok, img = cap.read()
    while ok:
        f1 += 1  # 1-based frame index
        cv2.putText(img, f"{vid_base} | frame {f1}", (10,25), font, 0.7, white, 2, cv2.LINE_AA)

        for a in frames_map.get(f1, []):
            pid  = a['pid']
            pts  = a.get('pts')
            bbox = a.get('bbox')

            # --- pred/true tại frame này
            pinfo = pred_map.get((pid, f1))
            pred  = pinfo['label'] if pinfo else 'Walk'
            score = pinfo['score'] if pinfo else None
            true  = gold_map.get((pid, f1), None)

            color = red if pred == 'Run' else green

            # 1) vẽ skeleton bằng helper
            if pts is not None:
                try:
                    img = draw_skeleton_auto(pts[:, :2], img)
                except Exception as e:
                    # log một lần cho frame này để biết lý do không vẽ
                    cv2.putText(img, f"SKEL ERR: {e}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)


            # 2) bbox màu đỏ/xanh (ưu tiên theo yêu cầu)
            if bbox is not None:
                x1,y1,x2,y2 = bbox
                pad = 4
                x1 = max(0, x1-pad); y1 = max(0, y1-pad)
                x2 = min(W-1, x2+pad); y2 = min(H-1, y2+pad)
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

                # (tuỳ chọn) dùng helper để in PID (nếu hàm helper cũng vẽ bbox, có thể bỏ rectangle phía trên)
                # if use_draw_pid:
                #     try:
                #         img = _draw_pid(img, [x1,y1,x2,y2], pid)
                #     except Exception:
                #         pass

                # label text
                lines = []
                # pred line
                lines.append(f"pid: {pid}")
                if score is not None:
                    lines.append(f"pred: {pred} ({score:.2f})")
                else:
                    lines.append(f"pred: {pred}")
                # true line (nếu không có gold thì hiển thị —)
                lines.append(f"true: {true if true is not None else '—'}")
                img = _draw_info_lines(img, [x1, y1, x2, y2], lines, size=18)
            
        writer.write(img)
        ok, img = cap.read()

    writer.release()
    cap.release()
    print(f"Saved: {out_path}")

def visualize_all_videos(pose_dir: str,
                         video_dir: str,
                         results_dir: str,
                         gold_csv_path: str,
                         out_dir: str = "viz_videos"):
    os.makedirs(out_dir, exist_ok=True)
    for fn in sorted(os.listdir(pose_dir)):
        if not fn.lower().endswith('.csv'):
            continue
        vid_base = canon_vid(os.path.splitext(fn)[0])
        pose_csv = os.path.join(pose_dir, fn)
        visualize_one_video(vid_base, video_dir, pose_csv, results_dir, gold_csv_path, out_dir)

if __name__ == "__main__":
    # visualize_one_video(vid_base = "Run_135-225_LA_Skirt_5.avi",
    #                         video_dir = "JPrecorded_video/JPRecord_all_rename",
    #                         pose_csv_path = "JPrecorded_track/run/Run_135-225_LA_Skirt_5.csv",
    #                         results_dir = "run_walking_results_csv",
    #                         gold_csv_path = "Toyota-running action.csv",
    #                         out_dir = "JPrecorded_visualization",
    #                         use_draw_pid = False)
    
    visualize_all_videos(
        pose_dir="JPrecorded_track/run",
        video_dir="JPrecorded_video/JPRecord_all_rename",
        results_dir="run_walking_results_csv",     # THƯ MỤC chứa các <vid>_eval.csv
        gold_csv_path="Toyota-running action.csv",
        out_dir="JPrecorded_visualization",
      
    )
    

