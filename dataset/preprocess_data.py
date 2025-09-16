import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from pathlib import Path
import cv2
import pickle
from tqdm import tqdm
from torchvision import transforms
import MinkowskiEngine as ME
import argparse

# Assuming constants.py exists and is correct
from constants import (
    IMG_MEAN, IMG_STD, MIN_BOX, MAX_BOX,
    NUM_PC_POINTS,T_HEAD_CAM,CAM_INTRINSIC 
)

def _sample_and_normalize_pcd(pcd: o3d.geometry.PointCloud, num_points: int) -> np.ndarray:
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)

    num_current_points = points.shape[0]
    if num_current_points == 0:
        return np.zeros((num_points, 6), dtype=np.float32)
    
    if num_current_points > num_points:
        indices = np.random.choice(num_current_points, num_points, replace=False)
    else:
        indices = np.random.choice(num_current_points, num_points, replace=True)
        
    sampled_points = points[indices]
    sampled_colors = colors[indices]

    sampled_colors = (sampled_colors - IMG_MEAN) / IMG_STD
    final_points = np.hstack([sampled_points, sampled_colors]).astype(np.float32)
    return final_points

def _create_single_view_pcd(
    rgb_image, depth_image, pose_head, pose_chassis, 
    intrinsics, extrinsics, bounding_box, num_points
    ) -> np.ndarray:
    K = np.array(CAM_INTRINSIC)
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(1280, 720, K[0,0], K[1,1], K[0,2], K[1,2])
    T_head_from_camera = T_HEAD_CAM
    
    o3d_rgb = o3d.geometry.Image(rgb_image)
    o3d_depth = o3d.geometry.Image(depth_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)

    pcd_cam.transform(T_head_from_camera)
    T_world_from_head = pose_to_matrix(pose_head)
    T_world_from_chassis = pose_to_matrix(pose_chassis)
    T_chassis_from_world = np.linalg.inv(T_world_from_chassis)
    T_chassis_from_head = T_chassis_from_world @ T_world_from_head
    pcd_cam.transform(T_chassis_from_head)

    pcd_cropped = pcd_cam.crop(bounding_box)
    return _sample_and_normalize_pcd(pcd_cropped, num_points)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    t = pose[:3]
    q = pose[3:]
    matrix = np.identity(4)
    matrix[:3, :3] = Rotation.from_quat(q).as_matrix()
    matrix[:3, 3] = t
    return matrix

def _voxelize_pcd_for_minkowski(pcd: o3d.geometry.PointCloud, voxel_size: float):
    if not pcd.has_points():
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float32)
    
    points = np.asarray(pcd.points, dtype=np.float32)
    feats = points.copy() 
    
    coords = np.floor(points / voxel_size).astype(np.int32)
    
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    
    return coords[unique_indices], feats[unique_indices]


def preprocess_episodes(root_dir: str, 
                        output_dir: str, 
                        voxel_size: float = 0.005,
                        use_multi_view: bool = False,):

    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bounding_box = o3d.geometry.AxisAlignedBoundingBox(MIN_BOX, MAX_BOX)

    pose_names = [
        'astribot_chassis', 'astribot_torso', 'astribot_arm_left', 
        'astribot_gripper_left', 'astribot_arm_right', 'astribot_gripper_right', 
        'astribot_head'
    ]
    image_names = ['head/rgb', 'left/rgb', 'right/rgb', 'torso/rgb']
    camera_configs = {
        'head': { 'robot_part_pose_key': 'astribot_head', 'intrinsics_key': 'Bolt', 'extrinsics_key': 'head_T_Bolt', 'rgb_path': '/images_dict/head/rgb', 'depth_path': '/images_dict/head/depth' },
    }
    
    episode_files = sorted(list(root_path.glob('*.hdf5')))

    for episode_path in tqdm(episode_files, desc="Processing Episodes"):
        output_file_path = output_path / episode_path.name

        if output_file_path.exists():
            print(f"Skipping {episode_path.name}, already processed.")
            continue

        with h5py.File(episode_path, 'r') as f_in, h5py.File(output_file_path, 'w') as f_out:
            num_timesteps = len(f_in['time'])
            
            if 'time' in f_in:
                f_out.create_dataset('time', data=f_in['time'][:])

            raw_poses_group = f_out.create_group('poses_dict')
            for name in pose_names:
                if f'/poses_dict/{name}' in f_in:
                    raw_poses_group.create_dataset(name, data=f_in[f'/poses_dict/{name}'][:])
            
            joints_path = '/joints_dict/joints_position_state'
            if joints_path in f_in:
                f_out.create_dataset(joints_path, data=f_in[joints_path][:])

            processed_pcd_coords_group = f_out.create_group('processed_pcd_coords')
            processed_pcd_feats_group = f_out.create_group('processed_pcd_feats')
            processed_images_group = f_out.create_group('processed_images')
            processed_sv_pcd_group = f_out.create_group('processed_single_view_pcd')

            for step_idx in tqdm(range(num_timesteps), desc=f"  Frames in {episode_path.name}", leave=False):
                current_poses = {name: f_in[f'/poses_dict/{name}'][step_idx] for name in pose_names}
                
                merged_pcd = o3d.geometry.PointCloud()
                T_world_from_chassis = pose_to_matrix(current_poses['astribot_chassis'])
                T_chassis_from_world = np.linalg.inv(T_world_from_chassis)
                
                for cam_name, config in camera_configs.items():
                    if not (config['rgb_path'] in f_in and config['depth_path'] in f_in): continue
                    rgb_image = f_in[config['rgb_path']][step_idx]
                    depth_image = f_in[config['depth_path']][step_idx]
                    K = np.array(CAM_INTRINSIC)
                    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(1280, 720, K[0,0], K[1,1], K[0,2], K[1,2])
                    T_part_from_camera = T_HEAD_CAM
                    o3d_rgb = o3d.geometry.Image(rgb_image)
                    o3d_depth = o3d.geometry.Image(depth_image)
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
                    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
                    pcd_cam.transform(T_part_from_camera)
                    part_pose_key = config['robot_part_pose_key']
                    T_world_from_part = pose_to_matrix(current_poses[part_pose_key])
                    T_chassis_from_part = T_chassis_from_world @ T_world_from_part
                    pcd_cam.transform(T_chassis_from_part)
                    merged_pcd += pcd_cam
                
                if merged_pcd.has_points():
                    merged_pcd_downsampled = merged_pcd.voxel_down_sample(voxel_size)
                    pcd_cropped = merged_pcd_downsampled.crop(bounding_box)
                else:
                    pcd_cropped = merged_pcd

                coords, feats = _voxelize_pcd_for_minkowski(pcd_cropped, voxel_size)
                processed_pcd_coords_group.create_dataset(str(step_idx), data=coords)
                processed_pcd_feats_group.create_dataset(str(step_idx), data=feats)

                head_rgb_path = '/images_dict/head/rgb'
                head_depth_path = '/images_dict/head/depth'
                if head_rgb_path in f_in and head_depth_path in f_in:
                    rgb_image = f_in[head_rgb_path][step_idx]
                    depth_image = f_in[head_depth_path][step_idx]
                    
                    sampled_pcd_data = _create_single_view_pcd(
                        rgb_image, depth_image, 
                        current_poses['astribot_head'], current_poses['astribot_chassis'],
                        CAM_INTRINSIC, T_HEAD_CAM, bounding_box, NUM_PC_POINTS
                    )
                    processed_sv_pcd_group.create_dataset(str(step_idx), data=sampled_pcd_data)

                img_group = processed_images_group.create_group(str(step_idx))
                for name in image_names:
                    key = f'/images_dict/{name}'
                    if key in f_in:
                        rgb_img = f_in[key][step_idx]
                        img_group.create_dataset(name.replace('/', '_'), data=rgb_img)


def calculate_joint_deltas_for_directory(processed_dir: str):
    """
    Iterates through all HDF5 files in a directory and converts the 
    '/joints_dict/joints_position_state' dataset from absolute positions to 
    deltas relative to the previous timestep. This is an in-place modification.

    Args:
        processed_dir (str): Directory containing the processed HDF5 files.

    used to calculate the delta of chassis movement
    """
    print(f"\nStarting joint delta calculation for directory: {processed_dir}")
    processed_path = Path(processed_dir)
    if not processed_path.is_dir():
        print(f"Error: Directory not found at {processed_dir}")
        return

    episode_files = sorted(list(processed_path.glob('*.hdf5')))
    joint_key = '/joints_dict/joints_position_state'

    for episode_path in tqdm(episode_files, desc="Calculating Joint Deltas"):
        try:
            with h5py.File(episode_path, 'r+') as f:
                if joint_key not in f:
                    continue

                original_joints = f[joint_key][:]
                
                if original_joints.shape[0] < 2:
                    delta_joints = np.zeros_like(original_joints)
                else:
                    delta_joints = np.zeros_like(original_joints)
                    delta_joints[1:] = original_joints[1:] - original_joints[:-1]
                
                del f[joint_key]
                f.create_dataset(joint_key, data=delta_joints, dtype=np.float32)

        except Exception as e:
            print(f"Error processing file {episode_path.name}: {e}")


if __name__ == '__main__':
    RAW_DATA_DIR = 'data/wine' 
    PROCESSED_DATA_DIR = 'data/processed_wine'

    parser = argparse.ArgumentParser(
        description="Preprocess robotics data episodes. Can run full pipeline or just calculate joint deltas."
    )
    parser.add_argument(
        '--raw_dir', 
        type=str, 
        default=RAW_DATA_DIR,
        help='Directory containing the raw HDF5 episode files.'
    )
    parser.add_argument(
        '--processed_dir', 
        type=str, 
        default=PROCESSED_DATA_DIR,
        help='Directory to save the processed HDF5 files.'
    )
    parser.add_argument(
        '--deltas_only',
        action='store_true',
        help='If set, only runs the joint delta calculation on an already processed directory specified by --processed_dir.'
    )
    args = parser.parse_args()

    if args.deltas_only:
        print("Running in 'deltas only' mode.")
        calculate_joint_deltas_for_directory(
            processed_dir=args.processed_dir
        )
        print("Joint delta calculation finished!")
    else:
        print("Starting full preprocessing pipeline...")
        preprocess_episodes(
            root_dir=args.raw_dir,
            output_dir=args.processed_dir,
            use_multi_view=False,
            voxel_size=0.005 
        )
        print("Initial preprocessing finished!")

        calculate_joint_deltas_for_directory(
            processed_dir=args.processed_dir
        )
        print("Full preprocessing pipeline finished!")