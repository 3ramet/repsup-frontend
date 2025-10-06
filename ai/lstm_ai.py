import pickle
from typing import Counter

from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def load_metadata(metadata_path):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        return metadata

all_dfs = []

def prep_data(points):
    ANGLE_COLS = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
              'left_hip', 'right_hip', 'left_knee', 'right_knee']
    global all_dfs
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    df = pd.DataFrame([points])
    df['source_file'] = 'real_data'
    all_dfs.append(df)
    all_dfs = all_dfs[-5:]  # เก็บแค่ 5 ไฟล์ล่าสุด

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # compute angles per row
    angles = []
    for index, row in combined_df.iterrows():
        try:
            landmarks = {i: {'x': row[f'x{i}'], 'y': row[f'y{i}']} for i in range(33)}
            p11=(landmarks[11]['x'], landmarks[11]['y']); p12=(landmarks[12]['x'], landmarks[12]['y'])
            p13=(landmarks[13]['x'], landmarks[13]['y']); p14=(landmarks[14]['x'], landmarks[14]['y'])
            p15=(landmarks[15]['x'], landmarks[15]['y']); p16=(landmarks[16]['x'], landmarks[16]['y'])
            p23=(landmarks[23]['x'], landmarks[23]['y']); p24=(landmarks[24]['x'], landmarks[24]['y'])
            p25=(landmarks[25]['x'], landmarks[25]['y']); p26=(landmarks[26]['x'], landmarks[26]['y'])
            p27=(landmarks[27]['x'], landmarks[27]['y']); p28=(landmarks[28]['x'], landmarks[28]['y'])

            angle_values = [
                calculate_angle(p11, p13, p15), calculate_angle(p12, p14, p16),
                calculate_angle(p13, p11, p23), calculate_angle(p14, p12, p24),
                calculate_angle(p11, p23, p25), calculate_angle(p12, p24, p26),
                calculate_angle(p23, p25, p27), calculate_angle(p24, p26, p28)
            ]
            angles.append(angle_values)
        except KeyError:
            angles.append([np.nan] * len(ANGLE_COLS))
            continue

    angle_df = pd.DataFrame(angles, columns=ANGLE_COLS)
    velocity_df = angle_df.groupby(combined_df['source_file']).diff().fillna(0)
    velocity_df.columns = [f'{col}_vel' for col in ANGLE_COLS]

    feature_df = pd.concat([combined_df.reset_index(drop=True), angle_df, velocity_df], axis=1)

    def assign_label(filename: str) -> int:
        if "_correct_" in filename or "_cor_" in filename: return 1
        elif "_incorrect_" in filename: return 0
        return -1

    feature_df['label'] = feature_df['source_file'].apply(assign_label)
    # print(f"{feature_df=}")
    feature_df.dropna(subset=ANGLE_COLS, inplace=True)
    feature_df = feature_df.reset_index(drop=True)
    # print(f"Shape after feature engineering and dropping NA in angles: {feature_df.shape}")
    return feature_df

def create_sequences(feature_df, source_series, window_size=5, step=1):
    """
    สร้างข้อมูลลำดับ 3 มิติ (samples, timesteps, features) สำหรับ LSTM
    """
    sequences = []
    win_sources = []
    idx_map = []
    
    temp_df = feature_df.copy()
    temp_df['source_file'] = source_series
    
    # จัดกลุ่มข้อมูลตาม source_file
    grouped = temp_df.groupby('source_file')
    
    for src, group_df in grouped:
        # ดึงเฉพาะ feature columns (ไม่เอา 'source_file' ที่เพิ่งเพิ่มเข้าไป)
        group_data = group_df.drop(columns=['source_file']).values
        
        # สร้าง sliding window (ส่วนที่เหลือเหมือนเดิม)
        for start in range(0, len(group_data) - window_size + 1, step):
            end = start + window_size
            window = group_data[start:end]
            sequences.append(window)
            win_sources.append(src)
            idx_map.append(list(group_df.index[start:end]))
            
    if not sequences:
        return None, None, None
        
    return np.array(sequences), win_sources, idx_map


bicep_curl_model = load_model("./ai/models/bicep_curl/lstm_model_final.keras")
lateral_raise_model = load_model("./ai/models/lateral_raise/lstm_model_final.keras")
squat_model = load_model("./ai/models/squat/lstm_model_final.keras")

def predict_exercise(points, metadata, model="bicep_curl"):
    if model == "bicep_curl":
        model_loaded = bicep_curl_model
    elif model == "lateral_raise":
        model_loaded = lateral_raise_model
    elif model == "squat":
        model_loaded = squat_model
    else:
        raise ValueError(f"Unknown model: {model}")

    scaler_loaded = metadata['scaler']
    best_thresh_loaded = metadata['best_threshold']
    feature_cols_loaded = metadata['feature_cols']
    WINDOW_SIZE_loaded = metadata['window_size']
    WINDOW_STEP = 1

    real_features = prep_data(points)
    if real_features is None:
        # print("No real test data found.")
        return ""

    real_labeled = real_features.copy().reset_index(drop=True)
    # print("Real test label counts (frame-level):", Counter(real_labeled['label']))

    # เลือก Feature columns ที่ใช้ตอนเทรน
    real_feature_df = real_labeled[feature_cols_loaded]
    real_sources_series = real_labeled['source_file']
    real_labels_series = real_labeled['label']

    print(f"{real_features=}")
    # # print(f"{real_labeled=}")
    # # print(f"{real_feature_df=}")

    # สร้าง Sequences 3 มิติจากข้อมูลทดสอบ
    X_real_sequences, real_win_sources, real_idx_map = create_sequences(
        real_feature_df, real_sources_series, window_size=WINDOW_SIZE_loaded, step=WINDOW_STEP
    )

    if X_real_sequences is None:
        # print("No windows in real test data.")
        exit(0)
        
    # --- 3. Scaling ข้อมูลทดสอบ (สำคัญมาก: ใช้ scaler ที่ fit ไว้แล้ว) ---
    n_samples, n_timesteps, n_features = X_real_sequences.shape
    X_real_reshaped = X_real_sequences.reshape(-1, n_features)
    X_real_scaled = scaler_loaded.transform(X_real_reshaped)
    X_real_scaled = X_real_scaled.reshape(n_samples, n_timesteps, n_features)


    # --- 4. ทำนายผล (Prediction) ---
    real_win_prob = model_loaded.predict(X_real_scaled).flatten()
    print(f"{real_win_prob=}")
    real_win_pred = (real_win_prob >= best_thresh_loaded).astype(int)


    # --- 5. การประเมินผล (Multi-level Evaluation) ---

    # --- Window-level evaluation ---
    # สร้าง Ground Truth (คำตอบจริง) ของแต่ละ Window
    real_window_labels = []
    for widx in real_idx_map:
        vals = real_labels_series.loc[widx]
        maj = int(vals.mode().iloc[0]) if not vals.mode().empty else int(vals.iloc[0])
        real_window_labels.append(maj)


    # print(f"{real_window_labels=}")
    # print(f"{real_win_pred=}")


    # --- Proper frame-level mapping ---
    n_frames = len(real_labeled)
    frame_prob_sums = np.zeros(n_frames, dtype=float)
    frame_vote_counts = np.zeros(n_frames, dtype=int)

    for win_i, indices in enumerate(real_idx_map):
        p = real_win_prob[win_i]
        for frame_idx in indices:
            # ตรวจสอบว่า frame_idx อยู่ในขอบเขตของ array หรือไม่
            if frame_idx < n_frames:
                frame_prob_sums[frame_idx] += p
                frame_vote_counts[frame_idx] += 1

    frame_mean_prob = np.zeros(n_frames, dtype=float)
    mask = frame_vote_counts > 0
    frame_mean_prob[mask] = frame_prob_sums[mask] / frame_vote_counts[mask]
    frame_pred = (frame_mean_prob >= best_thresh_loaded).astype(int)

    # ประเมินผลเฉพาะเฟรมที่มีการทำนาย (vote > 0)
    if False:
        if mask.sum() > 0:
            frame_true = real_labels_series.values
            # print("\nFrame-level report (for frames covered by windows):")
            # print(classification_report(frame_true[mask], frame_pred[mask], digits=4))
        else:
            pass
            # print("No frame-level votes could be computed.")

    # print("\n--- DONE ---")
    # นับจำนวนครั้งที่ทำนายได้เป็น 1 (bicep_curl)
    count_ones = np.sum(frame_pred)
    # print(f"Predicted bicep_curl count: {count_ones}")
    if count_ones > 0:
        return model
    else:
        return "unknown"
    