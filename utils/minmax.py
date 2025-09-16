import h5py
import numpy as np
from pathlib import Path
import sys
import random

DATA_DIRECTORIES = [
    'data/processed_wine',
]
SAMPLE_RATIO = 1

JOINT_STATE_PATH = '/joints_dict/joints_position_state'
JOINT_STATE_DIMS = 3

POSE_GROUP_PATH = '/poses_dict'
POSE_NAMES = [
     'astribot_torso', 'astribot_arm_left', 
    'astribot_gripper_left', 'astribot_arm_right', 'astribot_gripper_right', 
    'astribot_head'
]

def analyze_directories_for_action_bounds(directories: list, sample_ratio: float):
    print("--- Starting analysis ---")
    
    all_hdf5_files = []
    for directory in directories:
        path = Path(directory)
        if not path.is_dir():
            print(f"Warning: Directory does not exist, skipped -> {path}")
            continue
        found_files = list(path.rglob('*.hdf5'))
        print(f"Found {len(found_files)} .hdf5 files in '{path}'.")
        all_hdf5_files.extend(found_files)

    if not all_hdf5_files:
        print("Error: No .hdf5 files found in the specified directories. Please check DATA_DIRECTORIES.")
        sys.exit(1)
        
    num_total_files = len(all_hdf5_files)
    num_samples = max(1, int(num_total_files * sample_ratio))
    
    print(f"\nTotal number of files: {num_total_files}")
    print(f"Sampling {num_samples} files randomly for analysis ({sample_ratio:.0%} of total).")
    
    sampled_files = random.sample(all_hdf5_files, num_samples)
    
    all_action_vectors = []
    
    print("\nProcessing sampled files...")
    for i, file_path in enumerate(sampled_files):
        print(f"  ({i+1}/{num_samples}) Processing: {file_path.name}")
        try:
            with h5py.File(file_path, 'r') as f:
                if JOINT_STATE_PATH not in f:
                    print(f"    Warning: Missing '{JOINT_STATE_PATH}' in file, skipping.")
                    continue
                
                joints_data = f[JOINT_STATE_PATH][:]
                poses_data = {}
                missing_pose = False
                for name in POSE_NAMES:
                    pose_path = f"{POSE_GROUP_PATH}/{name}"
                    if pose_path not in f:
                        print(f"    Warning: Missing '{pose_path}' in file, skipping this file.")
                        missing_pose = True
                        break
                    poses_data[name] = f[pose_path][:]
                
                if missing_pose:
                    continue

                num_steps = len(joints_data)
                if num_steps == 0:
                    print("    Warning: File contains no timesteps, skipped.")
                    continue

                for step_idx in range(num_steps):
                    joint_part = joints_data[step_idx, :JOINT_STATE_DIMS]
                    pose_parts = [poses_data[name][step_idx] for name in POSE_NAMES]
                    full_action_vector = np.concatenate([joint_part] + pose_parts)
                    all_action_vectors.append(full_action_vector)

        except Exception as e:
            print(f"    Error occurred while processing {file_path.name}: {e}. Skipped.")
            continue

    if not all_action_vectors:
        print("\nError: No data successfully extracted from sampled files. Cannot compute bounds.")
        sys.exit(1)

    print("\nAll files processed. Computing statistics...")
    action_matrix = np.array(all_action_vectors)
    total_timesteps = action_matrix.shape[0]
    action_dim = action_matrix.shape[1]
    
    print(f"Analyzed {total_timesteps} timesteps in total.")
    
    lower_bounds = np.percentile(action_matrix, 5, axis=0)
    upper_bounds = np.percentile(action_matrix, 95, axis=0)
    
    lower_bounds_rounded = np.round(lower_bounds, 3)
    upper_bounds_rounded = np.round(upper_bounds, 3)

    print("\nAnalysis complete!")
    print(f"Constructed Action vector dimension: {action_dim}")
    print("\n" + "="*65)
    print("Copy the following results into your code to update MIN_POSE and MAX_POSE constants:")
    print("(These values correspond to the 5th and 95th percentiles of the data distribution, rounded to 3 decimals.)")
    print("="*65 + "\n")

    lower_str = np.array2string(lower_bounds_rounded, separator=', ', formatter={'float_kind': lambda x: f"{x:.3f}"})
    upper_str = np.array2string(upper_bounds_rounded, separator=', ', formatter={'float_kind': lambda x: f"{x:.3f}"})

    print(f"MIN_POSE = np.array({lower_str})")
    print(f"MAX_POSE = np.array({upper_str})")
    print("\n")


if __name__ == '__main__':
    for d in DATA_DIRECTORIES:
        Path(d).mkdir(exist_ok=True)
        
    analyze_directories_for_action_bounds(DATA_DIRECTORIES, SAMPLE_RATIO)
