import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import numpy as np

class WaveletFeatureExtractor:
    """
    Input:
    all_keypoints: list of N arrays (K×3) from MMPose
    all_bboxes:    list of N lists [x0, y0, w, h]
    Output:
    psd_features: dict of {feature_name: np.ndarray of shape (25,)}
    """
    def __init__(self, fs, wavelet='db1', dwt_level=3, save_dir=None):
        """
        kp_map: dict mapping semantic names to index in keypoints array
        wavelet: name of mother wavelet (for CWT if needed)
        fs: sampling frequency (frames per second)
        N: number of lowest-frequency PSD components to return
        """
        self.kp_map = {
            'head':0,'neck':1,
            'right_shoulder':2,'right_elbow':3,'right_wrist':4,
            'left_shoulder':5,'left_elbow':6,'left_wrist':7,
            'right_hip':8,'right_knee':9,'right_ankle':10,
            'left_hip':11,'left_knee':12,'left_ankle':13,
            'mid_hip':14
        }

        self.wavelet = wavelet
        self.fs = fs
        self.dwt_level = dwt_level

        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self.dist_pairs = [
            ('head','right_hip'), ('head','left_hip'),
            ('left_ankle','mid_hip'), ('right_ankle','mid_hip'),
            ('left_wrist','right_wrist')
        ]

        self.dist_feature_names = []
        for a,b in self.dist_pairs:
            self.dist_feature_names += [
                f"{a}→{b} Δx",
                f"{a}→{b} Δy"
            ]
        
        self.ang_triplets = [
            ('right_shoulder','right_elbow','right_wrist'),
            ('left_shoulder','left_elbow','left_wrist'),
            ('right_hip','right_knee','right_ankle'),
            ('left_hip','left_knee','left_ankle'),
            ('neck','left_hip','head'),
            ('right_shoulder','right_hip','right_knee'),
            ('left_shoulder','left_hip','left_knee'),
        ]
        self.ang_feature_names = [
            f"Angle at {j2}: {j1}-{j2}-{j3}"
            for j1,j2,j3 in self.ang_triplets
        ]

        # tên cho bbox-features
        self.bbox_feature_names = [
            "bbox width (normalized)",
            "bbox height (normalized)",
            "aspect ratio",
            "center speed"
        ]

    def _normalize_kp(self, keypoints, bbox):
        kpts = np.asarray(keypoints, dtype=float)
        x0, y0, bw, bh = bbox
        std = np.zeros((15,2), dtype=float)
        std[0] = (kpts[0,:2] - [x0,y0]) / [bw,bh]
        mid_sh = (kpts[5,:2] + kpts[6,:2]) * 0.5
        std[1] = (mid_sh - [x0,y0]) / [bw,bh]
        std[2] = (kpts[6,:2] - [x0,y0]) / [bw,bh]
        std[3] = (kpts[8,:2] - [x0,y0]) / [bw,bh]
        std[4] = (kpts[10,:2] - [x0,y0]) / [bw,bh]
        std[5] = (kpts[5,:2] - [x0,y0]) / [bw,bh]
        std[6] = (kpts[7,:2] - [x0,y0]) / [bw,bh]
        std[7] = (kpts[9,:2] - [x0,y0]) / [bw,bh]
        std[8] = (kpts[12,:2] - [x0,y0]) / [bw,bh]
        std[9] = (kpts[14,:2] - [x0,y0]) / [bw,bh]
        std[10] = (kpts[16,:2] - [x0,y0]) / [bw,bh]
        std[11] = (kpts[11,:2] - [x0,y0]) / [bw,bh]
        std[12] = (kpts[13,:2] - [x0,y0]) / [bw,bh]
        std[13] = (kpts[15,:2] - [x0,y0]) / [bw,bh]
        mid_hip = (kpts[11,:2] + kpts[12,:2]) * 0.5
        std[14] = (mid_hip - [x0,y0]) / [bw,bh]
        return std

    def _normalize_kp_new(self, keypoints, bbox):
        """
        keypoints: (K,2) đã scale về [0,1] theo frame (bạn đã làm ở trên)
        bbox: [x0,y0,w,h] cũng theo [0,1]
        Trả về std (15,2) theo thứ tự chuẩn hoá dưới đây để downstream dùng cố định.
        Nếu thiếu 'head'/'neck' ở K=14, sẽ fallback = mid_shoulder.
        """
        kpts = np.asarray(keypoints, dtype=float)
        x0, y0, bw, bh = bbox

        K = kpts.shape[0]
        idx = self._get_index_map_for_input_K(K)

        def P(name):
            i = idx.get(name, None)
            return None if i is None or i >= K else kpts[i, :2]

        # Vai/hông giữa
        L_sh, R_sh = P('left_shoulder'), P('right_shoulder')
        L_hp, R_hp = P('left_hip'), P('right_hip')

        mid_sh = None
        if L_sh is not None and R_sh is not None:
            mid_sh = 0.5*(L_sh + R_sh)
        elif P('neck') is not None:
            mid_sh = P('neck')
        else:
            # fallback: nếu chỉ có 1 vai
            mid_sh = L_sh if L_sh is not None else (R_sh if R_sh is not None else np.array([x0 + bw/2, y0 + bh*0.3]))

        mid_hip = None
        if L_hp is not None and R_hp is not None:
            mid_hip = 0.5*(L_hp + R_hp)
        else:
            # fallback: dùng knee nếu thiếu hip
            L_kn, R_kn = P('left_knee'), P('right_knee')
            if L_kn is not None and R_kn is not None:
                mid_hip = 0.5*(L_kn + R_kn)
            else:
                mid_hip = np.array([x0 + bw/2, y0 + bh*0.7])

        head = P('head')
        if head is None:
            # thiếu head: dùng mid_sh như trước
            head = mid_sh

        # Lấy các khớp còn lại (nếu thiếu, dùng mid_sh làm placeholder để không vỡ shape)
        def safe(name, default_point):
            p = P(name)
            return p if p is not None else default_point

        R_el, R_wr = safe('right_elbow', mid_sh), safe('right_wrist', mid_sh)
        L_el, L_wr = safe('left_elbow',  mid_sh), safe('left_wrist',  mid_sh)
        R_kn, R_an = safe('right_knee',  mid_hip), safe('right_ankle', mid_hip)
        L_kn, L_an = safe('left_knee',   mid_hip), safe('left_ankle',  mid_hip)

        # Chuẩn hoá về toạ độ tương đối bbox
        def norm(p):
            return (p - np.array([x0, y0])) / np.array([bw, bh])

        std = np.zeros((15, 2), dtype=float)
        # Thứ tự giữ NGUYÊN để downstream không đổi:
        # 0 head, 1 mid_shoulder, 2 R_sh, 3 R_el, 4 R_wr,
        # 5 L_sh, 6 L_el, 7 L_wr, 8 R_hip, 9 R_knee, 10 R_ank,
        # 11 L_hip, 12 L_knee, 13 L_ank, 14 mid_hip
        std[0]  = norm(head)
        std[1]  = norm(mid_sh)
        std[2]  = norm(safe('right_shoulder', mid_sh))
        std[3]  = norm(R_el)
        std[4]  = norm(R_wr)
        std[5]  = norm(safe('left_shoulder',  mid_sh))
        std[6]  = norm(L_el)
        std[7]  = norm(L_wr)
        std[8]  = norm(safe('right_hip',     mid_hip))
        std[9]  = norm(R_kn)
        std[10] = norm(R_an)
        std[11] = norm(safe('left_hip',      mid_hip))
        std[12] = norm(L_kn)
        std[13] = norm(L_an)
        std[14] = norm(mid_hip)
        return std
    def _get_index_map_for_input_K(self, K):
        """
        Trả về mapping tên->index theo layout của INPUT (trước khi chuẩn hoá).
        - Nếu K >= 17: giả định COCO-17 chuẩn.
        - Nếu K == 14: dùng 'simple14' (không có mắt/tai/mũi), bạn chỉnh ở đây nếu thứ tự khác.
        """
        if K >= 17:
            # COCO-17 (0..16)
            return {
                'head'           : 0,    # nose
                'left_shoulder'  : 5,
                'right_shoulder' : 6,
                'left_elbow'     : 7,
                'right_elbow'    : 8,
                'left_wrist'     : 9,
                'right_wrist'    : 10,
                'left_hip'       : 11,
                'right_hip'      : 12,
                'left_knee'      : 13,
                'right_knee'     : 14,
                'left_ankle'     : 15,
                'right_ankle'    : 16,
                # neck/mid_sh và mid_hip sẽ tính từ 2 vai/2 hông
            }
        elif K == 14:
            # SIMPLE-14 (ví dụ phổ biến). HÃY CHỈNH LẠI nếu thứ tự của bạn khác!
            # 0: head, 1: neck, 2: R_sh, 3: R_elb, 4: R_wri, 5: L_sh, 6: L_elb, 7: L_wri,
            # 8: R_hip, 9: R_knee, 10: R_ank, 11: L_hip, 12: L_knee, 13: L_ank
            return {
                'head'           : 0,
                'neck'           : 1,    # nếu không có, cứ xoá key 'neck'
                'right_shoulder' : 2,
                'right_elbow'    : 3,
                'right_wrist'    : 4,
                'left_shoulder'  : 5,
                'left_elbow'     : 6,
                'left_wrist'     : 7,
                'right_hip'      : 8,
                'right_knee'     : 9,
                'right_ankle'    : 10,
                'left_hip'       : 11,
                'left_knee'      : 12,
                'left_ankle'     : 13,
            }
        else:
            raise ValueError(f"Unsupported keypoint count K={K}. Provide a custom map.")

    def _normalize_bbox(self, bbox, frame_size):
        x, y, w, h = bbox
        fw, fh = frame_size
        return np.array([x/fw, y/fh, w/fw, h/fh], dtype=float)

    def extract_time_features(
        self,
        all_keypoints: np.ndarray, 
        all_bboxes: np.ndarray,     
        frame_size: tuple         
    ):
        """
        all_keypoints: list N × (K×3)
        all_bboxes:    list N × [x0,y0,w,h]
        """
        T = all_keypoints.shape[0]
        if T == 0:
            return None, None, None
        
        fw, fh = frame_size

        # --- 1) Normalize bbox theo frame và keypoints theo frame ---
        norm_bboxes = np.stack([
            self._normalize_bbox(b, frame_size)
            for b in all_bboxes
        ])  # (T,4)

        # keypoints normalized to [0,1] frame scale
        norm_kpts = np.stack([
            kps[:, :2] / np.array([fw, fh])
            for kps in all_keypoints
        ])  # (T, N_kp, 2)

        # --- 2) Normalize kp relative normalized bbox ---
        poses = np.stack([
            self._normalize_kp_new(norm_kpts[t], norm_bboxes[t])
            for t in range(T)
        ])  # (T, 15, 2)

        # --- 3) Tính distances từ poses ---
        dist_pairs = [
            ('head','right_hip'),('head','left_hip'),
            ('left_ankle','mid_hip'),
            ('right_ankle','mid_hip'),
            ('left_wrist','right_wrist')
        ]

        dist_arr = np.zeros((T, len(dist_pairs)*2))
        for t in range(T):
            vals = []
            for j1, j2 in dist_pairs:
                p1 = poses[t, self.kp_map[j1]]
                p2 = poses[t, self.kp_map[j2]]
                vals.extend((p1 - p2).tolist())
            dist_arr[t] = vals

        # --- 4) Tính angles từ poses ---
        ang_triplets = [
            ('right_shoulder','right_elbow','right_wrist'),
            ('left_shoulder','left_elbow','left_wrist'),
            ('right_hip','right_knee','right_ankle'),
            ('left_hip','left_knee','left_ankle'),
            ('neck','left_hip','head'),
            ('right_shoulder','right_hip','right_knee'),
            ('left_shoulder','left_hip','left_knee'),
        ]
        ang_arr = np.zeros((T, len(ang_triplets)))
        for t in range(T):
            angs = []
            for j1, j2, j3 in ang_triplets:
                v1 = poses[t, self.kp_map[j1]] - poses[t, self.kp_map[j2]]
                v2 = poses[t, self.kp_map[j3]] - poses[t, self.kp_map[j2]]
                mag = np.linalg.norm(v1) * np.linalg.norm(v2)
                dot = np.dot(v1, v2)
                angs.append(0.0 if mag == 0 else np.arccos(np.clip(dot/mag, -1, 1)))
            ang_arr[t] = angs

        # --- 5) Tính bbox-features từ normalized bbox ---
        bw = norm_bboxes[:,2]
        bh = norm_bboxes[:,3]
        aspect = np.divide(bw, bh, out=np.zeros_like(bw), where=bh>0)
        centers = norm_bboxes[:, :2] + norm_bboxes[:, 2:]/2

        # speed normalized by diag=√2
        diag = np.sqrt(1**2 + 1**2)
        speed = np.zeros(T)
        raw = np.linalg.norm(np.diff(centers, axis=0), axis=1) * self.fs
        speed[1:] = raw / diag
        bbox_feats = np.stack([bw, bh, aspect, speed], axis=1)

        return dist_arr, ang_arr, bbox_feats

    def compute_full_psd_freq(self, signal: np.ndarray):
        """
        Tính PSD đầy đủ và vector frequencies tương ứng.
        Trả về (freqs, psd_full).
        """
        n = len(signal)
        if n == 0:
            return np.array([]), np.array([])
        sig = signal - np.mean(signal)
        fft_vals = np.fft.rfft(sig)
        psd = (np.abs(fft_vals)**2) / n
        freqs = np.fft.rfftfreq(n, d=1/self.fs)
        return freqs, psd

    def compute_psd_n(self, signal: np.ndarray, n: int = 25):
        """
        Tính PSD và giữ n giá trị đầu.
        """
        _, psd = self.compute_full_psd_freq(signal)
        length = len(psd)
        if length < n:
            return np.pad(psd, (0, n-length))
        return psd[:n]
    
    def extract_psd_signals(
        self,
        all_keypoints: np.ndarray,
        all_bboxes: np.ndarray,
        frame_size: tuple
    ):
        """
        Tính PSD đầy đủ và frequencies cho từng feature.
        Trả về dict feature_name -> (freqs, psd_full)
        """
        dist_arr, ang_arr, bbox_feats = self.extract_time_features(all_keypoints, all_bboxes, frame_size)
        if dist_arr is None:
            return {}
        psd_dict={}
        for i in range(dist_arr.shape[1]):
            freqs, psd = self.compute_full_psd_freq(dist_arr[:,i])
            psd_dict[f"dist_{i}"] = (freqs, psd)
        for i in range(ang_arr.shape[1]):
            freqs, psd = self.compute_full_psd_freq(ang_arr[:,i])
            psd_dict[f"ang_{i}"] = (freqs, psd)
        for i in range(bbox_feats.shape[1]):
            freqs, psd = self.compute_full_psd_freq(bbox_feats[:,i])
            psd_dict[f"bbox_{i}"] = (freqs, psd)
        return psd_dict
    
    def extract_freq_features(
        self,
        all_keypoints: np.ndarray,
        all_bboxes: np.ndarray,
        frame_size: tuple,
        n: int = 25
    ):
        """
        Trả về vector PSD_n (n giá trị đầu mỗi feature)
        """
        psd_dict = self.extract_psd_signals(all_keypoints, all_bboxes, frame_size)
        vecs=[]
        for freqs, psd in psd_dict.values():
            vecs.append(self.compute_psd_n(psd, n))
        return np.concatenate(vecs)


    def compute_dwt_coeffs(self, signal):
        coeffs = pywt.wavedec(signal - np.mean(signal), self.wavelet, level=self.dwt_level)
        feats = []
        # for c in coeffs:
        #     feats.extend([
        #         np.sum(c**2),
        #         np.mean(c),
        #         np.std(c),
        #         np.max(np.abs(c)),
        #     ])
        # return np.array(feats)
        names = [f'cA{self.dwt_level}'] + [f'cD{l}' for l in range(self.dwt_level, 0, -1)]
        return {name: coeff for name, coeff in zip(names, coeffs)}

    def extract_dwt_features(self, all_keypoints, all_bboxes, frame_size):
        """
        Trả về nested dict:
        {
           pid_feature_name: { 'cA4': [...], 'cD4': [...], … }
        }
        """
        dist_arr, ang_arr, bbox_feats = self.extract_time_features(
            all_keypoints, all_bboxes, frame_size)
        if dist_arr is None:
            return {}

        result = {}
        arrays = {'dist': dist_arr, 'ang': ang_arr, 'bbox': bbox_feats}
        for feat_name, arr in arrays.items():
            for idx in range(arr.shape[1]):
                sig = arr[:, idx]
                coeff_dict = self.compute_dwt_coeffs(sig)
                result[f'{feat_name}_{idx}'] = coeff_dict
        return result

    def _save_and_close(self, fig, name):
        if self.save_dir:
            path = os.path.join(self.save_dir, f"{name}.png")
            fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def plot_time_series(self, signal: np.ndarray, title: str):
        fig, ax = plt.subplots()
        ax.plot(signal)
        ax.set(title=f"Time series – {title}", xlabel="Frame index", ylabel="Value")
        ax.grid(True)
        self._save_and_close(fig, f"time_{title}")
    
    def plot_dwt_time_series(self, signal: np.ndarray, title: str):
        coeffs = pywt.wavedec(signal - np.mean(signal), self.wavelet, level=self.dwt_level)
        names = [f'cA{self.dwt_level}'] + [f'cD{l}' for l in range(self.dwt_level, 0, -1)]
        fig, ax = plt.subplots()
        for name, c in zip(names, coeffs):
            ax.plot(c, label=name)
        ax.set(title=f"DWT Coefficients – {title}", xlabel="Index", ylabel="Amplitude")
        ax.legend()
        ax.grid(True)
        self._save_and_close(fig, f"dwt_all_{title}")


    def plot_dwt_energy_bar(self, signal: np.ndarray, title: str):
        coeffs = pywt.wavedec(signal - np.mean(signal), self.wavelet, level=self.dwt_level)
        names = [f'cA{self.dwt_level}'] + [f'cD{l}' for l in range(self.dwt_level, 0, -1)]
        energies = [np.sum(c**2) for c in coeffs]
        fig, ax = plt.subplots()
        ax.bar(names, energies)
        ax.set(title=f"DWT energy per level – {title}", xlabel="Level", ylabel="Energy")
        ax.grid(axis='y')
        self._save_and_close(fig, f"energy_{title}")    
    
    def plot_scalogram(self, signal: np.ndarray, title: str):
        scales = np.arange(1, 64)
        try:
            wobj = pywt.ContinuousWavelet(self.wavelet)
            if not getattr(wobj, 'complex_cwt', False):
                wobj = pywt.ContinuousWavelet('morl')
        except Exception:
            wobj = pywt.ContinuousWavelet('morl')
        coef, freqs = pywt.cwt(signal - np.mean(signal), scales, wobj,
                               sampling_period=1/self.fs)
        power = np.abs(coef)**2
        fig, ax = plt.subplots(figsize=(8,4))
        im = ax.imshow(power, aspect='auto', origin='lower',
                       extent=[0, len(signal), freqs[0], freqs[-1]])
        fig.colorbar(im, ax=ax, label='Power')
        ax.set(title=f"Scalogram (CWT) – {title}", xlabel="Time (frames)", ylabel="Frequency (Hz)")
        self._save_and_close(fig, f"scalogram_{title}.png")

    def compute_dominant_frequency(self, signal: np.ndarray):
        scales = np.arange(1, 64)
        try:
            wobj = pywt.ContinuousWavelet(self.wavelet)
            if not getattr(wobj, 'complex_cwt', False):
                wobj = pywt.ContinuousWavelet('morl')
        except Exception:
            wobj = pywt.ContinuousWavelet('morl')
        coef, freqs = pywt.cwt(signal - np.mean(signal), scales, wobj,
                               sampling_period=1/self.fs)
        power = np.abs(coef)**2
        energy_per_scale = power.sum(axis=1)
        idx = np.argmax(energy_per_scale)
        dom_scale = scales[idx]
        dom_freq = pywt.scale2frequency(wobj, dom_scale) * self.fs
        return dom_freq, energy_per_scale[idx]

    def plot_dominant_frequency(self, signal: np.ndarray, title: str):
        scales = np.arange(1, 64)
        try:
            wobj = pywt.ContinuousWavelet(self.wavelet)
            if not getattr(wobj, 'complex_cwt', False):
                wobj = pywt.ContinuousWavelet('morl')
        except Exception:
            wobj = pywt.ContinuousWavelet('morl')
        coef, freqs = pywt.cwt(signal - np.mean(signal), scales, wobj,
                               sampling_period=1/self.fs)
        power = np.abs(coef)**2
        energy_per_scale = power.sum(axis=1)
        freq_vector = pywt.scale2frequency(wobj, scales) * self.fs
        fig, ax = plt.subplots()
        ax.plot(freq_vector, energy_per_scale)
        ax.set(title=f"Wavelet Energy Spectrum – {title}", xlabel="Frequency (Hz)", ylabel="Energy")
        ax.grid(True)
        dom_freq, dom_energy = self.compute_dominant_frequency(signal)
        ax.scatter([dom_freq], [dom_energy], color='red', label=f"Dominant ≈ {dom_freq:.2f} Hz")
        ax.legend()
        self._save_and_close(fig, f"domfreq_{title}.png")

    def plot_all_features(self, dist_arr: np.ndarray, ang_arr: np.ndarray, bbox_feats: np.ndarray):
        # Distances
        for i in range(dist_arr.shape[1]):
            desc = self.dist_feature_names[i]
            title = f"dist_{i}: {desc}"
            sig = dist_arr[:, i]
            self.plot_time_series(sig, title)
            self.plot_dwt_time_series(sig, title)
            self.plot_dwt_energy_bar(sig, title)
            self.plot_scalogram(sig, title)

        # Angles
        for j in range(ang_arr.shape[1]):
            desc = self.ang_feature_names[j]
            title = f"ang_{j}: {desc}"
            sig = dist_arr[:, j]
            self.plot_time_series(sig, title)
            self.plot_dwt_time_series(sig, title)
            self.plot_dwt_energy_bar(sig, title)
            self.plot_scalogram(sig, title)
            # self.plot_dominant_frequency(sig, title)

        # Bbox features
        for k in range(bbox_feats.shape[1]):
            desc = self.bbox_feature_names[k]
            title = f"bbox_{k}: {desc}"
            sig = bbox_feats[:, k]
            self.plot_time_series(sig, title)
            self.plot_dwt_time_series(sig, title)
            self.plot_dwt_energy_bar(sig, title)
            self.plot_scalogram(sig, title)
            # self.plot_dominant_frequency(sig, title)
        
    def extract_wr_frame_features(
        self,
        all_keypoints,   
        all_bboxes,      
        frame_size, 
    ):
        """
        Trả về:
        feats: (T, 10) theo thứ tự:
            [ cos_knee_L, cos_knee_R,
            hip_flex_L, hip_flex_R,
            arm_swing_L, arm_swing_R,
            ankleLR_over_torso,
            wristLR_over_shoulder,
            midhip_y_local_norm,
            speed ]
        feat_names: list tên tương ứng
        """

        eps = 1e-12
        fw, fh = frame_size
        T = int(all_keypoints.shape[0])
        if T == 0:
            return None, None

        # 1) Bbox → [0,1] theo frame
        norm_bboxes = np.stack([self._normalize_bbox(b, frame_size) for b in all_bboxes], axis=0)  # (T,4)
        centers = norm_bboxes[:, :2] + norm_bboxes[:, 2:] / 2.0                                   # (T,2)

        # 2) Speed của bbox center (chuẩn hoá theo đường chéo đơn vị sqrt(2))
        fs = float(getattr(self, "fs", 30.0))
        speed = np.zeros(T, dtype=float)
        if T >= 2:
            diag = np.sqrt(2.0)
            raw = np.linalg.norm(np.diff(centers, axis=0), axis=1) * fs
            speed[1:] = raw / diag

        # 3) Keypoints → [0,1] theo frame
        if all_keypoints.shape[-1] >= 2:
            norm_kpts = np.stack([kps[:, :2] / np.array([fw, fh], dtype=float) for kps in all_keypoints], axis=0)
        else:
            raise ValueError("all_keypoints must have at least 2 channels (x,y).")

        # 4) Chuẩn hoá keypoints theo bbox → (T, 15, 2) (đúng thứ tự bạn đang dùng)
        poses = np.stack([self._normalize_kp_new(norm_kpts[t], norm_bboxes[t]) for t in range(T)], axis=0)

        idx = self.kp_map  # dùng đúng mapping bạn đã có

        # Bảo đảm các key cần thiết tồn tại
        required = ['head','neck','right_shoulder','right_elbow','right_wrist',
                    'left_shoulder','left_elbow','left_wrist',
                    'right_hip','right_knee','right_ankle',
                    'left_hip','left_knee','left_ankle','mid_hip']

        for k in required:
            if k not in idx:
                raise KeyError(f"kp_map thiếu key: {k}")

        def unit(v):
            n = np.linalg.norm(v)
            return v / (n + eps)

        def cos_between(u, v):
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < eps or nv < eps:
                return 0.0
            return float(np.dot(u, v) / (nu * nv))

        feat_names = [
            "cos_knee_L", "cos_knee_R",
            "hip_flex_L", "hip_flex_R",
            "arm_swing_L", "arm_swing_R",
            "ankleLR_over_torso",
            "wristLR_over_shoulder",
            "midhip_y_local_norm",
            "speed",
        ]
        feats = np.zeros((T, len(feat_names)), dtype=float)

        for t in range(T):
            J = poses[t]  # (15,2), toạ độ đã bbox-normalized

            # Lấy toạ độ các khớp cần
            LHip, RHip = J[idx['left_hip']],   J[idx['right_hip']]
            LKnee, RKnee = J[idx['left_knee']], J[idx['right_knee']]
            LAnk, RAnk = J[idx['left_ankle']], J[idx['right_ankle']]
            LSh,  RSh  = J[idx['left_shoulder']], J[idx['right_shoulder']]
            LWr,  RWr  = J[idx['left_wrist']],    J[idx['right_wrist']]
            midHip     = J[idx['mid_hip']]
            midSh      = J[idx['neck']]  # 'neck' trong kp_map của bạn chính là mid-shoulder

            # Trục thân (local Y) và độ dài chuẩn hoá
            body_axis = unit(midSh - midHip)                   # Y_local
            torso_len = np.linalg.norm(midSh - midHip) + eps
            shoulder_width = np.linalg.norm(RSh - LSh) + eps

            # 1) cos_knee_L/R: cos góc tại đầu gối
            cos_knee_L = cos_between(LHip - LKnee, LAnk - LKnee)
            cos_knee_R = cos_between(RHip - RKnee, RAnk - RKnee)

            # 2) hip_flex_*: cos giữa (hip→knee) và trục thân
            hip_flex_L = cos_between(LKnee - LHip, body_axis)
            hip_flex_R = cos_between(RKnee - RHip, body_axis)

            # 3) arm_swing_*: cos giữa (shoulder→wrist) và trục thân
            arm_swing_L = cos_between(LWr - LSh, body_axis)
            arm_swing_R = cos_between(RWr - RSh, body_axis)

            # 4) ankleLR_over_torso: ||LAnk - RAnk|| / torso_len
            ankleLR_over_torso = np.linalg.norm(LAnk - RAnk) / torso_len

            # 5) wristLR_over_shoulder: ||LWr - RWr|| / shoulder_width
            wristLR_over_shoulder = np.linalg.norm(LWr - RWr) / shoulder_width

            # 6) midhip_y_local_norm: chiếu mid_hip vào trục thân, gốc = tâm bbox (0.5,0.5), chia torso_len
            bbox_center = np.array([0.5, 0.5], dtype=float)
            midhip_y_local_norm = float(np.dot(midHip - bbox_center, body_axis) / torso_len)

            feats[t, :] = [
                cos_knee_L, cos_knee_R,
                hip_flex_L, hip_flex_R,
                arm_swing_L, arm_swing_R,
                ankleLR_over_torso,
                wristLR_over_shoulder,
                midhip_y_local_norm,
                speed[t],
            ]

        return feats, feat_names
