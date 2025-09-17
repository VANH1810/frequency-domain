import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.fft import rfft, rfftfreq

# --- 0. Auxiliary Functions ---
def calculate_vector_angle(vec1, vec2):
    """Calculates the angle in degrees between two 2D vectors."""
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 < 1e-6 or norm_vec2 < 1e-6:
        return np.nan
    vec1_u = vec1 / norm_vec1
    vec2_u = vec2 / norm_vec2
    dot_product = np.dot(vec1_u, vec2_u)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)

# --- 1. Individual Feature Calculation Functions ---

def calculate_movement_speed(pose_stream, j_indices, fps):
    """Calculates the normalized movement speed based on hip center displacement."""
    if pose_stream.shape[0] < 2: return np.nan
    neck_pos = pose_stream[:, j_indices['neck'], :]
    hip_centers = (pose_stream[:, j_indices['lhip'], :] + pose_stream[:, j_indices['rhip'], :]) / 2
    torso_lengths = np.linalg.norm(neck_pos - hip_centers, axis=1)
    torso_lengths[torso_lengths < 1e-6] = 1.0
    
    pixel_displacements = np.linalg.norm(hip_centers[1:] - hip_centers[:-1], axis=1)
    avg_torso_for_disp = (torso_lengths[1:] + torso_lengths[:-1]) / 2
    normalized_displacements = pixel_displacements / avg_torso_for_disp
    
    total_time = (pose_stream.shape[0] - 1) / fps
    total_normalized_distance = np.sum(normalized_displacements)
    return total_normalized_distance / total_time if total_time > 0 else np.nan

def calculate_head_angles(pose_stream, j_indices):
    """Calculates the average head angle relative to the vertical axis and the torso."""
    if pose_stream.shape[0] < 1: return np.nan, np.nan
    neck_pos = pose_stream[:, j_indices['neck'], :]
    nose_pos = pose_stream[:, j_indices['nose'], :]
    hip_centers = (pose_stream[:, j_indices['lhip'], :] + pose_stream[:, j_indices['rhip'], :]) / 2
    
    head_vecs = nose_pos - neck_pos
    torso_vecs = neck_pos - hip_centers
    vertical_ref_vec = np.array([0, -1])

    angles_vert = [calculate_vector_angle(hv, vertical_ref_vec) for hv in head_vecs if np.linalg.norm(hv) > 1e-6]
    angles_torso = [calculate_vector_angle(hv, tv) for hv, tv in zip(head_vecs, torso_vecs) if np.linalg.norm(hv) > 1e-6 and np.linalg.norm(tv) > 1e-6]
    
    return np.nanmean(angles_vert), np.nanmean(angles_torso)

def calculate_torso_angle(pose_stream, j_indices):
    """Calculates the average torso lean angle relative to the vertical axis."""
    if pose_stream.shape[0] < 1: return np.nan
    neck_pos = pose_stream[:, j_indices['neck'], :]
    hip_centers = (pose_stream[:, j_indices['lhip'], :] + pose_stream[:, j_indices['rhip'], :]) / 2
    torso_vecs = neck_pos - hip_centers
    vertical_ref_vec = np.array([0, -1])
    
    angles = [calculate_vector_angle(tv, vertical_ref_vec) for tv in torso_vecs if np.linalg.norm(tv) > 1e-6]
    return np.nanmean(angles)

def calculate_shoulder_metrics(pose_stream, j_indices):
    """Calculates normalized shoulder width and shrugging level."""
    if pose_stream.shape[0] < 1: return np.nan, np.nan
    neck_pos = pose_stream[:, j_indices['neck'], :]
    lshoulder_pos, rshoulder_pos = pose_stream[:, j_indices['lshoulder'], :], pose_stream[:, j_indices['rshoulder'], :]
    hip_centers = (pose_stream[:, j_indices['lhip'], :] + pose_stream[:, j_indices['rhip'], :]) / 2
    torso_lengths = np.linalg.norm(neck_pos - hip_centers, axis=1)
    torso_lengths[torso_lengths < 1e-6] = 1.0

    shoulder_dist = np.linalg.norm(rshoulder_pos - lshoulder_pos, axis=1)
    norm_width = np.mean(shoulder_dist / torso_lengths)
    
    shoulder_center_y = (lshoulder_pos[:, 1] + rshoulder_pos[:, 1]) / 2
    norm_shrug = np.mean((neck_pos[:, 1] - shoulder_center_y) / torso_lengths)
    
    return norm_width, norm_shrug

def calculate_arm_swing_coordination_symmetry(pose_stream, j_indices):
    """Calculates arm swing coordination (Pearson correlation) and symmetry (amplitude ratio)."""
    if pose_stream.shape[0] < 2: return np.nan, np.nan
    lwrist_pos, rwrist_pos = pose_stream[:, j_indices['lwrist'], :], pose_stream[:, j_indices['rwrist'], :]
    hip_centers_x = (pose_stream[:, j_indices['lhip'], 0] + pose_stream[:, j_indices['rhip'], 0]) / 2
    
    lwrist_x_rel = lwrist_pos[:, 0] - hip_centers_x
    rwrist_x_rel = rwrist_pos[:, 0] - hip_centers_x
    
    if len(lwrist_x_rel) < 2 or np.std(lwrist_x_rel) < 1e-6 or np.std(rwrist_x_rel) < 1e-6:
        coordination = np.nan
    else:
        coordination, _ = pearsonr(lwrist_x_rel, rwrist_x_rel)
        
    lwrist_amp = np.max(lwrist_x_rel) - np.min(lwrist_x_rel)
    rwrist_amp = np.max(rwrist_x_rel) - np.min(rwrist_x_rel)
    
    if rwrist_amp > 1e-6:
        symmetry = lwrist_amp / rwrist_amp
    elif lwrist_amp < 1e-6:
        symmetry = 1.0
    else:
        symmetry = np.nan
        
    return coordination, symmetry

def calculate_arm_swing_speed_frequency(pose_stream, j_indices, fps):
    """Calculates average arm swing speed and dominant frequency for both arms."""
    if pose_stream.shape[0] < 2: return np.nan, np.nan
    
    # Common variables
    neck_pos = pose_stream[:, j_indices['neck'], :]
    hip_centers = (pose_stream[:, j_indices['lhip'], :] + pose_stream[:, j_indices['rhip'], :]) / 2
    avg_torso_length = np.mean(np.linalg.norm(neck_pos - hip_centers, axis=1))

    # Helper function for single arm
    def _get_speed_freq(wrist_idx, shoulder_idx):
        wrist_pos = pose_stream[:, wrist_idx, :]
        shoulder_pos = pose_stream[:, shoulder_idx, :]
        
        # Speed
        wrist_vel = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1) * fps
        norm_speed = np.mean(wrist_vel) / avg_torso_length if avg_torso_length > 1e-6 else np.nan
        
        # Frequency
        swing_signal_x = wrist_pos[:, 0] - shoulder_pos[:, 0]
        N = len(swing_signal_x)
        freq = np.nan
        if N > 1 and np.std(swing_signal_x) > 1e-3:
            yf = rfft(swing_signal_x - np.mean(swing_signal_x))
            xf = rfftfreq(N, 1/fps)
            if len(xf) > 1:
                freq = xf[np.argmax(np.abs(yf[1:])) + 1]
        return norm_speed, freq

    lspeed, lfreq = _get_speed_freq(j_indices['lwrist'], j_indices['lshoulder'])
    rspeed, rfreq = _get_speed_freq(j_indices['rwrist'], j_indices['rshoulder'])

    avg_speed = np.nanmean([lspeed, rspeed])
    avg_freq = np.nanmean([lfreq, rfreq])
    
    return avg_speed, avg_freq

def calculate_cadence_and_steplength(pose_stream, j_indices, fps, normalized_body_speed):
    """Calculates cadence (steps/min) and an approximated normalized step length."""
    if pose_stream.shape[0] < fps / 2: return np.nan, np.nan
    
    ankle_y_vel = np.diff(pose_stream[:, j_indices['lankle'], 1]) * fps
    cadence_hz = np.nan
    if len(ankle_y_vel) > 1 and np.std(ankle_y_vel) > 1e-3:
        N = len(ankle_y_vel)
        yf = rfft(ankle_y_vel - np.mean(ankle_y_vel))
        xf = rfftfreq(N, 1/fps)
        valid_indices = np.where((xf >= 0.5) & (xf <= 3.0))[0]
        if len(valid_indices) > 0:
            peak_idx = valid_indices[np.argmax(np.abs(yf[valid_indices]))]
            cadence_hz = xf[peak_idx] * 2

    cadence_spm = cadence_hz * 60 if pd.notna(cadence_hz) else np.nan
    step_length = normalized_body_speed / cadence_hz if pd.notna(cadence_hz) and cadence_hz > 0 else np.nan
    
    return cadence_spm, step_length

def calculate_knee_angles(pose_stream, j_indices):
    """Calculates average knee flexion angle and range of motion for both knees."""
    if pose_stream.shape[0] < 1: return np.nan, np.nan
    
    def _get_angles(hip_idx, knee_idx, ankle_idx):
        hip_pos, knee_pos, ankle_pos = pose_stream[:, hip_idx, :], pose_stream[:, knee_idx, :], pose_stream[:, ankle_idx, :]
        vec_kh, vec_ka = hip_pos - knee_pos, ankle_pos - knee_pos
        return [calculate_vector_angle(v1, v2) for v1, v2 in zip(vec_kh, vec_ka) if pd.notna(calculate_vector_angle(v1, v2))]

    lknee_angles = _get_angles(j_indices['lhip'], j_indices['lknee'], j_indices['lankle'])
    rknee_angles = _get_angles(j_indices['rhip'], j_indices['rknee'], j_indices['rankle'])
    
    all_angles = lknee_angles + rknee_angles
    if not all_angles: return np.nan, np.nan
    
    avg_angle = np.nanmean(all_angles)
    rom_angle = np.nanmax(all_angles) - np.nanmin(all_angles) if len(all_angles) > 1 else 0
    return avg_angle, rom_angle

def calculate_vertical_oscillation(pose_stream, j_indices):
    """Calculates the normalized vertical oscillation of the hip center."""
    if pose_stream.shape[0] < 2: return np.nan
    hip_centers = (pose_stream[:, j_indices['lhip'], :] + pose_stream[:, j_indices['rhip'], :]) / 2
    neck_pos = pose_stream[:, j_indices['neck'], :]
    avg_torso_length = np.mean(np.linalg.norm(neck_pos - hip_centers, axis=1))

    if avg_torso_length < 1e-6: return np.nan
    
    hip_y_oscillation = np.max(hip_centers[:, 1]) - np.min(hip_centers[:, 1])
    return hip_y_oscillation / avg_torso_length

def calculate_jerk(pose_stream, j_indices, fps):
    """Calculates the average jerk of the hip center."""
    if pose_stream.shape[0] < 4: return np.nan
    center_positions = (pose_stream[:, j_indices['lhip'], :] + pose_stream[:, j_indices['rhip'], :]) / 2
    
    if center_positions.shape[0] < 4: return np.nan
    dt = 1.0 / fps
    velocity = np.diff(center_positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    
    return np.mean(np.linalg.norm(jerk, axis=1)) if jerk.shape[0] > 0 else np.nan

def calculate_kinetic_energy_proxy(pose_stream, fps):
    """Calculates a proxy for kinetic energy based on the sum of squared velocities."""
    if pose_stream.shape[0] < 2: return np.nan
    velocities = np.diff(pose_stream, axis=0) * fps
    speed_sq = np.sum(velocities**2, axis=2)
    return np.mean(np.sum(speed_sq, axis=1)) if speed_sq.shape[0] > 0 else np.nan


# --- 2. Main Coordinating Function ---
def calculate_all_gait_features(pose_stream, j_indices, fps=30):
    """
    Receives a single skeleton time-series and calculates all 11 feature categories.
    
    Returns:
        dict: A dictionary containing all calculated features.
    """
    if pose_stream is None or pose_stream.shape[0] < 4:
        # Create a dictionary with all feature keys mapped to NaN
        return {key: np.nan for key in ALL_FEATURE_KEYS}

    features = {}

    # Calculate features using the modular functions
    features['normalized_speed'] = calculate_movement_speed(pose_stream, j_indices, fps)
    features['head_angle_vertical'], features['head_angle_torso'] = calculate_head_angles(pose_stream, j_indices)
    features['torso_angle_vertical'] = calculate_torso_angle(pose_stream, j_indices)
    features['norm_shoulder_width'], features['norm_shrugging_level'] = calculate_shoulder_metrics(pose_stream, j_indices)
    features['arm_swing_coordination'], features['arm_swing_symmetry_ratio'] = calculate_arm_swing_coordination_symmetry(pose_stream, j_indices)
    features['Avg_arm_norm_speed'], features['Avg_arm_dom_freq'] = calculate_arm_swing_speed_frequency(pose_stream, j_indices, fps)
    
    # Cadence and Step Length calculation depends on normalized_speed
    cadence, step_length = calculate_cadence_and_steplength(pose_stream, j_indices, fps, features.get('normalized_speed', np.nan))
    features['cadence_spm'] = cadence
    features['norm_step_length'] = step_length
    
    features['Avg_knee_avg_angle'], features['Avg_knee_rom_angle'] = calculate_knee_angles(pose_stream, j_indices)
    features['norm_vertical_oscillation'] = calculate_vertical_oscillation(pose_stream, j_indices)
    features['avg_hip_jerk'] = calculate_jerk(pose_stream, j_indices, fps)
    features['kinetic_energy_proxy'] = calculate_kinetic_energy_proxy(pose_stream, fps)

    return features

# --- 3. Main Processing Logic ---
# --- Configuration ---
CSV_FILE_PATH = 'your_base_data.csv' 
OUTPUT_CSV_PATH = 'df_all_features_calculated.csv'

# !!! CRITICAL: Modify the joint indices according to your skeleton format !!!
JOINT_INDICES = {
    'nose': 0, 'neck': 1, 'rshoulder': 2, 'relbow': 3, 'rwrist': 4,
    'lshoulder': 5, 'lelbow': 6, 'lwrist': 7, 'rhip': 8, 'rknee': 9,
    'rankle': 10, 'lhip': 11, 'lknee': 12, 'lankle': 13
}
NUM_JOINTS = 14
FPS = 30
# --- End of Configuration ---

# List of all feature keys for creating NaN dictionaries
ALL_FEATURE_KEYS = [
    'normalized_speed', 'head_angle_vertical', 'head_angle_torso', 'torso_angle_vertical',
    'norm_shoulder_width', 'norm_shrugging_level', 'arm_swing_coordination',
    'arm_swing_symmetry_ratio', 'Avg_arm_norm_speed', 'Avg_arm_dom_freq',
    'cadence_spm', 'norm_step_length', 'Avg_knee_avg_angle',
    'Avg_knee_rom_angle', 'norm_vertical_oscillation', 'avg_hip_jerk',
    'kinetic_energy_proxy'
]

# Dummy data creation for standalone execution
if not os.path.exists(CSV_FILE_PATH):
    print(f"'{CSV_FILE_PATH}' not found. Creating dummy data...")
    data_size = 50
    df_rows = []
    pose_data_dir = './pose_data/' 
    os.makedirs(pose_data_dir, exist_ok=True)
    for i in range(data_size):
        pose_data = np.random.rand(np.random.randint(60,120), NUM_JOINTS, 2) * 100
        full_path = os.path.abspath(os.path.join(pose_data_dir, f"sample_pose_all_feat_{i}.npy"))
        np.save(full_path, pose_data)
        df_rows.append({'pose': full_path, 'true_label': np.random.randint(0,2)})
    df_base = pd.DataFrame(df_rows)
    df_base.to_csv(CSV_FILE_PATH, index=False)
    print("Dummy data created.")

try:
    df_base = pd.read_csv(CSV_FILE_PATH)
    print(f"Successfully loaded '{CSV_FILE_PATH}' with {len(df_base)} entries.")
except FileNotFoundError:
    print(f"Error: CSV file '{CSV_FILE_PATH}' not found.")
    exit()

all_features_list = []

print("Starting batch feature calculation for all samples...")
for index, row in tqdm(df_base.iterrows(), total=df_base.shape[0]):
    file_path = row['pose']
    features = { 'pose': file_path } # Start with the pose path for merging
    try:
        pose_data = np.load(file_path)
        if pose_data.ndim != 3 or pose_data.shape[1] != NUM_JOINTS or pose_data.shape[2] != 2:
            print(f"\nWarning: Skipping file {os.path.basename(file_path)} due to incorrect shape.")
            features.update({key: np.nan for key in ALL_FEATURE_KEYS})
        else:
            # Call the main coordinating function
            calculated_features = calculate_all_gait_features(pose_data, JOINT_INDICES, fps=FPS)
            features.update(calculated_features)

    except FileNotFoundError:
        print(f"\nWarning: File not found at {file_path}.")
        features.update({key: np.nan for key in ALL_FEATURE_KEYS})
    except Exception as e:
        print(f"\nError processing file {os.path.basename(file_path)}: {e}")
        features.update({key: np.nan for key in ALL_FEATURE_KEYS})
    
    all_features_list.append(features)

# Convert list of dictionaries to DataFrame
df_features = pd.DataFrame(all_features_list)

# Merge the calculated features back into the original DataFrame
df_final = pd.merge(df_base, df_features, on='pose', how='left')

print("\n--- Calculation complete. DataFrame with all features generated. ---")
print(df_final.head())
print("\nDataFrame Shape:", df_final.shape)
print("\nColumns:", df_final.columns.tolist())

# Save the final DataFrame
df_final.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nDataFrame with all calculated features has been saved to '{OUTPUT_CSV_PATH}'")