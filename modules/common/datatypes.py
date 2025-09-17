import numpy as np
from collections import deque
from modules.common.constants import MAX_SEQ_LEN



class TrackObj:
    __slots__ = ('missed_frames', 'max_seq', 'frm_idx', 'pid', 'box', 'kp',
                 'seq_box', 'seq_kp', 'seq_frm', 'is_full_sequence')

    def __init__(self, frm_idx, pid, box, kp):
        self.missed_frames = 0
        self.max_seq = MAX_SEQ_LEN
        self.frm_idx = frm_idx  # last
        self.pid = pid  
        self.box = box  # 4x1 # last
        self.kp = kp  # 18x3 # last 
        self.seq_box = deque(maxlen=self.max_seq)  # MAX_LENx4x1 - history box
        self.seq_kp = deque(maxlen=self.max_seq)  # MAX_LENx18x3 - history keypoint
        self.seq_frm = deque(maxlen=self.max_seq) # MAX_LENx1 - history frame
        self.is_full_sequence = False
    
    def update(self, new_frm, new_box, new_kp):
        self.frm_idx = new_frm
        self.box = new_box
        self.kp = new_kp
        self.seq_frm.append(new_frm)
        self.seq_box.append(new_box)  
        self.seq_kp.append(new_kp)
        
    def get_seq_kp_2d(self, kp_score=0.05, seq_len=20):
        """
        Get the last `seq_len` keypoints with shape (seq_len, 18, 2), filtering out keypoints with confidence < kp_score.
        """
        if not self.seq_kp:
            return np.zeros((seq_len, 18, 2), dtype=np.float32)

        # Convert deque to numpy array
        seq_kp_array = np.array(self.seq_kp)  # Shape: (max_seq, 18, 3)

        if seq_kp_array.shape[0] >= seq_len:
            self.is_full_sequence = True
            seq_kp_array = seq_kp_array[-seq_len:]  
        else:
            self.is_full_sequence = False
            padding = np.zeros((seq_len - seq_kp_array.shape[0], 18, 3), dtype=np.float32)
            seq_kp_array = np.vstack((padding, seq_kp_array))

        # Lọc ra keypoints có confidence < kp_score
        mask = seq_kp_array[:, :, 2] < kp_score
        seq_kp_array[mask, :2] = 0  # (x, y) = 0 if keypoint <  confidence

        return seq_kp_array[:, :, :2]  
    
    def get_seq_box(self):
        return self.seq_box

    def get_seq_kp(self):
        return self.kp
    
    
    def to_dict(self):
        return {
            "frm_idx": self.frm_idx,
            "pid": self.pid,
            "box": self.box.tolist() if isinstance(self.box, np.ndarray) else self.box,  
            "kp": self.kp.tolist() if isinstance(self.kp, np.ndarray) else self.kp,  
            "seq_box": [b.tolist() if isinstance(b, np.ndarray) else b for b in self.seq_box],  
            "seq_kp": [kp.tolist() if isinstance(kp, np.ndarray) else kp for kp in self.seq_kp],
            "seq_frm": np.array(self.seq_frm).tolist(),
        }

class ActionObj:
    __slots__ = (
        "missed_frames", "frm_idx", "max_seq", "track_obj", "action", 
        "seq_action", "final_vote"
    )

    def __init__(self, frm_idx, track_obj, action): 
        self.missed_frames = 0
        self.frm_idx = frm_idx 
        self.max_seq = MAX_SEQ_LEN
        self.track_obj = track_obj
        self.action = action #last 
        self.seq_action = deque(maxlen=MAX_SEQ_LEN) 
        self.final_vote = None
        
    def update(self, new_action, new_trk_obj=None):
        self.action = new_action
        self.seq_action.append(new_action)
        if new_trk_obj is not None:
            self.track_obj = new_trk_obj
        
        
    def get_seq_action(self):
        return self.seq_action

    def get_final_vote(self):
        return self.final_vote
    
    
    def to_dict(self):
        return {
            "frm_idx": self.frm_idx,
            "action": self.action,
            "seq_action": list(self.seq_action),  
            "final_vote": self.final_vote,
            "track_obj": self.track_obj.to_dict() 
        }