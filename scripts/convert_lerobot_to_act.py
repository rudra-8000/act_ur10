import os
import h5py
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

LEROBOT_DATASET_DIR = '/home_local/rudra_1/rudra/data/grasp_place_v21_backup'
OUTPUT_DIR = '/home_local/rudra_1/rudra/data/grasp_place_v21_hdf5'
CAMERA_NAMES = ['cam_high', 'cam_right_wrist']
IMG_SIZE = (480, 640)  # H, W — match your camera resolution

def get_video_frames(video_path, num_frames):
    """Decode all frames from an MP4 video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads BGR, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE[1], IMG_SIZE[0]))
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)  # [T, H, W, 3]

# def convert_episode(episode_idx, parquet_path, video_dir, output_dir):
#     df = pd.read_parquet(parquet_path)
    
#     # --- Extract qpos and action ---
#     # Adjust column names based on your actual parquet schema
#     # For UR10 with PincOpen gripper: 6 arm joints + 1 gripper = 7D
#     state_cols = [c for c in df.columns if 'observation.state' in c]
#     action_cols = [c for c in df.columns if c == 'action' or 'action.' in c]
    
#     if state_cols:
#         # If stored as separate columns
#         qpos = df[state_cols].values.astype(np.float32)
#     else:
#         # If stored as a single column with arrays
#         qpos = np.stack(df['observation.state'].values).astype(np.float32)
    
#     if len(action_cols) > 1:
#         action = df[action_cols].values.astype(np.float32)
#     else:
#         action = np.stack(df['action'].values).astype(np.float32)
    
#     T = len(df)
#     state_dim = qpos.shape[1]
    
#     # qvel: not available from LeRobot typically, use zeros
#     qvel = np.zeros_like(qpos)
    
#     # --- Load video frames ---
#     images = {}
#     for cam_name in CAMERA_NAMES:
#         ep_str = f'episode_{episode_idx:06d}'
#         video_path = Path(video_dir) / f'observation.images.{cam_name}' / f'{ep_str}.mp4'
#         frames = get_video_frames(video_path, T)
#         # Trim or pad to match T
#         if len(frames) > T:
#             frames = frames[:T]
#         elif len(frames) < T:
#             pad = np.zeros((T - len(frames), *frames.shape[1:]), dtype=np.uint8)
#             frames = np.concatenate([frames, pad], axis=0)
#         images[cam_name] = frames
    
#     # --- Write HDF5 ---
#     os.makedirs(output_dir, exist_ok=True)
#     out_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
#     with h5py.File(out_path, 'w') as f:
#         f.attrs['sim'] = False
#         obs = f.create_group('observations')
#         obs.create_dataset('qpos', data=qpos)
#         obs.create_dataset('qvel', data=qvel)
#         img_grp = obs.create_group('images')
#         for cam_name, frames in images.items():
#             img_grp.create_dataset(
#                 cam_name, data=frames,
#                 chunks=(1, IMG_SIZE[0], IMG_SIZE[1], 3),
#                 compression='lzf'  # fast compression
#             )
#         f.create_dataset('action', data=action)
    
#     print(f'Wrote episode {episode_idx}: T={T}, state_dim={state_dim}')

def convert_episode(episode_idx, parquet_path, video_dir, output_dir):
    df = pd.read_parquet(parquet_path)

    # Both 'action' and 'observation.state' are object columns
    # where each cell contains a numpy array or list — must use np.stack
    qpos   = np.stack(df['observation.state'].values).astype(np.float32)  # [T, 7]
    action = np.stack(df['action'].values).astype(np.float32)              # [T, 7]

    T         = len(df)
    state_dim = qpos.shape[1]
    qvel      = np.zeros_like(qpos)

    # --- Load video frames ---
    images = {}
    for cam_name in CAMERA_NAMES:
        ep_str     = f'episode_{episode_idx:06d}'
        video_path = Path(video_dir) / f'observation.images.{cam_name}' / f'{ep_str}.mp4'
        frames     = get_video_frames(video_path, T)
        if len(frames) > T:
            frames = frames[:T]
        elif len(frames) < T:
            pad    = np.zeros((T - len(frames), *frames.shape[1:]), dtype=np.uint8)
            frames = np.concatenate([frames, pad], axis=0)
        images[cam_name] = frames

    # --- Write HDF5 ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
    with h5py.File(out_path, 'w') as f:
        f.attrs['sim'] = False
        obs = f.create_group('observations')
        obs.create_dataset('qpos',  data=qpos)
        obs.create_dataset('qvel',  data=qvel)
        img_grp = obs.create_group('images')
        for cam_name, frames in images.items():
            img_grp.create_dataset(
                cam_name, data=frames,
                chunks=(1, IMG_SIZE[0], IMG_SIZE[1], 3),
                compression='lzf'
            )
        f.create_dataset('action', data=action)

    print(f'Wrote episode {episode_idx}: T={T}, state_dim={state_dim}')

if __name__ == '__main__':
    data_dir = Path(LEROBOT_DATASET_DIR) / 'data' / 'chunk-000'
    video_dir = Path(LEROBOT_DATASET_DIR) / 'videos' / 'chunk-000'
    
    parquet_files = sorted(data_dir.glob('episode_*.parquet'))
    for i, pq_path in enumerate(tqdm(parquet_files)):
        convert_episode(i, pq_path, video_dir, OUTPUT_DIR)
    
    print(f'Done. {len(parquet_files)} episodes written to {OUTPUT_DIR}')