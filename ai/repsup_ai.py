import os
import pickle
import numpy as np
import pandas as pd
import glob
from typing import List, Optional, Tuple, Any

def load_stacking_model(model_path: str) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with open(model_path, "rb") as file:
        stacking_model: dict = pickle.load(file)
        print(f"Model loaded from {model_path}")
        print(f"{type(stacking_model)=}, keys: {list(stacking_model.keys())}")
    return stacking_model

JOINT_ANGLE_COLUMNS = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
              'left_hip', 'right_hip', 'left_knee', 'right_knee']

def calculate_joint_angle(point_a: Tuple[float, float], point_b: Tuple[float, float], point_c: Tuple[float, float]) -> float:
    array_a = np.array(point_a)
    array_b = np.array(point_b)
    array_c = np.array(point_c)
    radians = np.arctan2(array_c[1] - array_b[1], array_c[0] - array_b[0]) - np.arctan2(array_a[1] - array_b[1], array_a[0] - array_b[0])
    angle_degrees = np.abs(radians * 180.0 / np.pi)
    if angle_degrees > 180.0:
        angle_degrees = 360.0 - angle_degrees
    return angle_degrees

def read_landmark_csv_files(directory_path: str, filename_patterns: List[str]) -> Optional[List[pd.DataFrame]]:
    print(f"\n--- Loading data from: {directory_path} ---")
    dataframes: List[pd.DataFrame] = []
    for filename_pattern in filename_patterns:
        for csv_file_path in glob.glob(os.path.join(directory_path, filename_pattern)):
            try:
                dataframe = pd.read_csv(csv_file_path)
            except Exception as error:
                print(f"  Could not read {csv_file_path}: {error}")
                continue
            print(f"  Read: {os.path.basename(csv_file_path)}")
            dataframe['source_file'] = os.path.basename(csv_file_path)
            dataframes.append(dataframe)
    if not dataframes:
        print(f"!!! WARNING: No files found in {directory_path} with given patterns.")
        return None
    return dataframes

def extract_joint_angles_from_landmarks(dataframe: pd.DataFrame) -> pd.DataFrame:
    joint_angle_rows: List[List[float]] = []
    for _, row in dataframe.iterrows():
        try:
            landmarks = {i: {'x': row[f'x{i}'], 'y': row[f'y{i}']} for i in range(33)}
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            left_elbow = (landmarks[13]['x'], landmarks[13]['y'])
            right_elbow = (landmarks[14]['x'], landmarks[14]['y'])
            left_wrist = (landmarks[15]['x'], landmarks[15]['y'])
            right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
            left_hip = (landmarks[23]['x'], landmarks[23]['y'])
            right_hip = (landmarks[24]['x'], landmarks[24]['y'])
            left_knee = (landmarks[25]['x'], landmarks[25]['y'])
            right_knee = (landmarks[26]['x'], landmarks[26]['y'])
            left_ankle = (landmarks[27]['x'], landmarks[27]['y'])
            right_ankle = (landmarks[28]['x'], landmarks[28]['y'])
            joint_angles = [
                calculate_joint_angle(left_shoulder, left_elbow, left_wrist),
                calculate_joint_angle(right_shoulder, right_elbow, right_wrist),
                calculate_joint_angle(left_elbow, left_shoulder, left_hip),
                calculate_joint_angle(right_elbow, right_shoulder, right_hip),
                calculate_joint_angle(left_shoulder, left_hip, left_knee),
                calculate_joint_angle(right_shoulder, right_hip, right_knee),
                calculate_joint_angle(left_hip, left_knee, left_ankle),
                calculate_joint_angle(right_hip, right_knee, right_ankle)
            ]
            joint_angle_rows.append(joint_angles)
        except KeyError:
            joint_angle_rows.append([np.nan] * len(JOINT_ANGLE_COLUMNS))
            continue
    return pd.DataFrame(joint_angle_rows, columns=JOINT_ANGLE_COLUMNS)

def create_feature_dataframe(landmark_dataframe: pd.DataFrame, joint_angle_dataframe: pd.DataFrame) -> pd.DataFrame:
    joint_angle_velocity_dataframe = joint_angle_dataframe.groupby(landmark_dataframe['source_file']).diff().fillna(0)
    joint_angle_velocity_dataframe.columns = [f'{col}_vel' for col in JOINT_ANGLE_COLUMNS]
    print(f"{joint_angle_velocity_dataframe=}")
    feature_dataframe = pd.concat([landmark_dataframe.reset_index(drop=True), joint_angle_dataframe, joint_angle_velocity_dataframe], axis=1)
    feature_dataframe.dropna(subset=JOINT_ANGLE_COLUMNS, inplace=True)
    feature_dataframe = feature_dataframe.reset_index(drop=True)
    print(f"Shape after feature engineering: {feature_dataframe.shape}")
    return feature_dataframe

def load_and_prepare_features(directory_path: str, filename_patterns: List[str]) -> Optional[pd.DataFrame]:
    landmark_dataframes = read_landmark_csv_files(directory_path, filename_patterns)
    if landmark_dataframes is None:
        return None
    combined_landmark_dataframe = pd.concat(landmark_dataframes, ignore_index=True)
    joint_angle_dataframe = extract_joint_angles_from_landmarks(combined_landmark_dataframe)
    feature_dataframe = create_feature_dataframe(combined_landmark_dataframe, joint_angle_dataframe)
    return feature_dataframe

def generate_windowed_features(
    joint_angle_dataframe: pd.DataFrame,
    source_files: pd.Series,
    window_size: int = 5,
    window_step: int = 1
) -> Tuple[Optional[pd.DataFrame], Optional[List[Any]]]:
    windowed_feature_series_list: List[pd.Series] = []
    windowed_source_file_list: List[str] = []
    window_index_map: List[List[int]] = []
    source_file_groups: dict = {}
    for index, source_file in enumerate(source_files):
        source_file_groups.setdefault(source_file, []).append(index)

    for source_file, indices in source_file_groups.items():
        if len(indices) == 0: 
            continue
        for start in range(0, max(1, len(indices) - window_size + 1), window_step):
            window_indices = indices[start:start+window_size]
            window_block = joint_angle_dataframe.loc[window_indices]
            window_mean = window_block.mean(axis=0).add_prefix('win_mean_')
            window_std = window_block.std(axis=0).fillna(0).add_prefix('win_std_')
            window_features = pd.concat([window_mean, window_std])
            windowed_feature_series_list.append(window_features)
            windowed_source_file_list.append(source_file)
            window_index_map.append(window_indices)
        if len(indices) < window_size:
            window_block = joint_angle_dataframe.loc[indices]
            window_mean = window_block.mean(axis=0).add_prefix('win_mean_')
            window_std = window_block.std(axis=0).fillna(0).add_prefix('win_std_')
            window_features = pd.concat([window_mean, window_std])
            windowed_feature_series_list.append(window_features)
            windowed_source_file_list.append(source_file)
            window_index_map.append(indices)
    if not windowed_feature_series_list:
        return None, None
    windowed_feature_dataframe = pd.DataFrame(windowed_feature_series_list).reset_index(drop=True)
    return windowed_feature_dataframe, window_index_map

memory: list[dict[str, float]] = []
count: int = 0

def predict_exercise(points: dict[str, float], model: dict) -> Optional[str]:
    xgb_model = model['model']
    scaler_model = model['scaler']
    best_threshold: float = float(model.get('best_threshold', 0.5))

    global memory, count
    points['source_file'] = 'input.csv'
    count += 1
    
    n = 5
    length = len(memory)
    last_n_landmark = []
    if length >= n:
        last_n_landmark = memory[-n:]
        memory = memory[-(n-1):]
    elif length > 0:
        last_n_landmark = memory
    
    landmark_df = pd.DataFrame([points])

    for last_n in last_n_landmark:
        landmark_df = pd.concat([landmark_df, pd.DataFrame([last_n])], ignore_index=True)
    memory.append(points)

    joint_angle_df: pd.DataFrame = extract_joint_angles_from_landmarks(landmark_df)
    feature_df: pd.DataFrame = create_feature_dataframe(landmark_df, joint_angle_df)
    # update memory

    Xw: Optional[pd.DataFrame]
    idx_map: Optional[list]
    Xw, idx_map = generate_windowed_features(
        joint_angle_dataframe=feature_df[JOINT_ANGLE_COLUMNS],
        source_files=feature_df['source_file'],
        window_size=n,
        window_step=1
    )

    print(f"Generated windowed features: {Xw.shape if Xw is not None else None}")

    X_scaled: Optional[np.ndarray] = None
    if Xw is not None:
        X_scaled = scaler_model.transform(Xw)
        print(f"{Xw.shape=}, {X_scaled.shape=}")
        predictions_proba: np.ndarray = xgb_model.predict_proba(X_scaled)[:, 1]
        predictions: np.ndarray = (predictions_proba >= best_threshold).astype(int)
        print(f"Predictions (proba >= {best_threshold}): {predictions_proba}")
        print(f"Predictions (class): {predictions}")
        if np.any(predictions == 1):
            return "bicep_curl"
        else:
            return "unknown"
        
    return None

