import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
FP_RATE = 5

def time_to_frame(t_sec, fps=FP_RATE, mode='floor'):
    return int(np.floor(t_sec*fps)) if mode=='floor' else int(np.ceil(t_sec*fps))

def load_gt_from_csv(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=['Pid'])
    df['time_start'] = pd.to_timedelta(df['Time start']).dt.total_seconds()
    df['time_end']   = pd.to_timedelta(df['Time end']).dt.total_seconds()
    df['vid']        = df['Name video'].astype(str).str.strip()
    df['pid_str']    = df['Pid'].astype(int).astype(str)
    df['f0'] = df['time_start'].apply(lambda t: time_to_frame(t, FP_RATE, 'floor'))
    df['f1'] = df['time_end'].apply(lambda t: time_to_frame(t, FP_RATE, 'ceil'))
    mask = df['f1'] <= df['f0']
    df.loc[mask, 'f1'] = df.loc[mask, 'f0'] + FP_RATE

    gt = {}
    for _, r in df.iterrows():
        key = (r.vid, r.pid_str)
        gt.setdefault(key, []).append((r.f0, r.f1, r.Action))
    return gt

# 1) Load tất cả ground-truth
gt_v1 = load_gt_from_csv('Toyota_data - client_v1.csv')
gt_v2 = load_gt_from_csv('Toyota_data - client_v2.csv')
gt_v3 = load_gt_from_csv('Toyota_data - client_v3.csv')
gt_v4 = load_gt_from_csv('Toyota_data - client_v4.csv')
gt_all = {**gt_v2}

group_map = {
    "Fall_down":  "Falling",
    "Staggering": "Staggering"
}
def map_group(label):
    return group_map.get(label, "Other")

pred_root = 'merged_predictions'
eval_root = 'client_v1_merge_evaluation'
os.makedirs(eval_root, exist_ok=True)

all_records = []   # <-- gom toàn bộ record ở đây

# 1) Đọc prediction, lưu eval CSV per-VID, và append vào all_records
for dirpath, _, files in os.walk(pred_root):
    for fname in files:
        if not fname.endswith('_predictions.csv'):
            continue
        vid = fname.replace('_predictions.csv','')
        dfp = pd.read_csv(os.path.join(dirpath, fname))

        # records chỉ cho per-file (để xuất eval CSV)
        records = []
        for _, row in dfp.iterrows():
            pid = str(row['PID'])
            s_p, e_p   = map(int, row['Frame_Range'].split('-'))
            pred_group = map_group(row['Label'])
            pred_score = row['Prob']

            intervals = gt_v1.get((vid, pid), [])
            best_act, best_ov = None, 0
            for f0, f1, act in intervals:
                ist = max(s_p, f0)
                ied = min(e_p, f1-1)
                ov  = max(0, ied - ist + 1)
                if ov > best_ov:
                    best_ov, best_act = ov, act

            true_group = best_act if best_act else "Other"

            rec = {
                "Frame_Range": f"{s_p}-{e_p}",
                "VID": vid,
                "PID": pid,
                "True": true_group,
                "Pred": pred_group,
                "Score": pred_score
            }
            records.append(rec)
            all_records.append(rec)   # <-- gom vào all_records

        # Xuất file eval CSV cho video này
        eval_df = pd.DataFrame(records)
        out_path = os.path.join(eval_root, f"{vid}_eval.csv")
        eval_df.to_csv(out_path, index=False)
        print(f"Saved evaluation for {vid} → {out_path}")



# 2) Sau khi xử lý xong tất cả videos, tính metrics trên all_records

dfr = pd.DataFrame(all_records)

to_force = []
for vid, grp in dfr.groupby('VID'):
    has_true    = (grp['True'] == "Falling").any()
    has_pred    = (grp['Pred'] == "Falling").any()
    if has_true and has_pred:
        to_force.append(vid)

# Chỉ gán lại Pred="Falling" cho các record của những video này **và** có True=="Falling"  
mask = dfr['VID'].isin(to_force) & (dfr['True'] == "Falling")
dfr.loc[mask, 'Pred'] = "Falling"


y_true = dfr['True']
y_pred = dfr['Pred']

labels = ["Falling", "Staggering", "Other"]
print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

print("Per-class metrics:")
print(classification_report(
    y_true, y_pred, labels=labels, digits=4
))