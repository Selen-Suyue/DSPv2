import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
import MinkowskiEngine as ME
import torchvision.transforms as T
import warnings
from torchvision import transforms
from PIL import Image
from dataset.constants import IMG_MEAN,IMG_STD
from dataset.load_pose import load_pose_limits

NUM_DEMO = 100
NAN_FILL_VALUE = 0.0

def collate_fn(batch):
    collated_batch = {}
    pcd_coords_list = []
    pcd_feats_list = []
    
    for item in batch:
        if item['pcd_coords'].shape[0] > 0:
            pcd_coords_list.append(item['pcd_coords'])
            pcd_feats_list.append(item['pcd_feats'])
        
    if pcd_coords_list:
        batched_pcd_coords, batched_pcd_feats = ME.utils.sparse_collate(
            [torch.from_numpy(c) for c in pcd_coords_list], 
            [torch.from_numpy(f) for f in pcd_feats_list], 
            dtype=torch.float32
        )
        collated_batch['pcd_coords'] = batched_pcd_coords
        collated_batch['pcd_feats'] = batched_pcd_feats
    else: 
        collated_batch['pcd_coords'] = torch.empty(0, 4, dtype=torch.int32)
        collated_batch['pcd_feats'] = torch.empty(0, 3, dtype=torch.float32)

    if 'images' in batch[0] and batch[0]['images']:
        image_keys = sorted(batch[0]['images'].keys())
        images_per_sample = [
                torch.stack([item['images'][key] for key in image_keys]) 
                for item in batch
            ]
        collated_batch['images'] = torch.stack(images_per_sample)
    
    collated_batch['pose'] = torch.stack([item['pose'] for item in batch])
    collated_batch['action'] = torch.stack([item['action'] for item in batch])
    
    return collated_batch


class FastMinkSet(Dataset):
    def __init__(self, 
                 processed_root_dir: str, 
                 action_horizon: int = 16,
                aug_jitter_params: list = [0.4, 0.4, 0.2, 0.1],  
                aug_jitter_prob: float = 0.3,  
                task: str = None
                 ):
        
        self.root_dir = Path(processed_root_dir)
        self.Ta = action_horizon
        self.pose_names = [
            'astribot_chassis', 'astribot_torso', 'astribot_arm_left', 
            'astribot_gripper_left', 'astribot_arm_right', 'astribot_gripper_right', 
            'astribot_head'
        ]
        self.image_names = ['head_rgb', 'left_rgb', 'right_rgb', 'torso_rgb']
        self.jitter = T.Compose([
            T.RandomPerspective(distortion_scale=0.2, p=0.5), 
            T.RandomResizedCrop((224, 224), scale=(0.6, 1.0), ratio=(16/9, 16/9)),
            T.RandomApply([
                T.ColorJitter(
                    brightness=aug_jitter_params[0],
                    contrast=aug_jitter_params[1],
                    saturation=aug_jitter_params[2],
                    hue=aug_jitter_params[3]
                )
            ], p=aug_jitter_prob)
        ])
        MIN_POSE, MAX_POSE = load_pose_limits(task)
        self.norm_min = torch.from_numpy(MIN_POSE.astype(np.float32)).float()
        self.norm_max = torch.from_numpy(MAX_POSE.astype(np.float32)).float()
        self.norm_range = (self.norm_max - self.norm_min).float()
        self.norm_range[self.norm_range == 0] = 1e-8
        self.image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ])
        self.episode_files = sorted(list(self.root_dir.glob('*.hdf5')))[:NUM_DEMO]
        self.episode_lookup = []
        for i, file_path in enumerate(self.episode_files):
            with h5py.File(file_path, 'r') as f:
                num_timesteps = len(f['time'])
                if num_timesteps > self.Ta:
                    for step_idx in range(num_timesteps - self.Ta):
                        self.episode_lookup.append((i, step_idx))
        
        if not self.episode_lookup:
            raise RuntimeError(f"In '{self.root_dir}' no valid episodes found. ")

    def normalize_pose(self, pose_tensor):
        return (pose_tensor - self.norm_min) / self.norm_range

    def __len__(self):
        return len(self.episode_lookup)

    def __getitem__(self, idx):
        episode_idx, step_idx = self.episode_lookup[idx]
        file_path = self.episode_files[episode_idx]
        
        with h5py.File(file_path, 'r') as f:
            pcd_coords = f[f'/processed_pcd_coords/{step_idx}'][:]
            pcd_feats = f[f'/processed_pcd_feats/{step_idx}'][:]

            images = {}
            for name in self.image_names:
                key = f'/processed_images/{step_idx}/{name}'
                if key in f:
                    image = self.jitter(Image.fromarray(f[key][:]))
                    processed_image = self.image_transform(image)
                    if torch.isnan(processed_image).any():
                        warning_message = (
                            f"Warning: Image '{name}' at step {step_idx} contains NaN values. "
                            f"Filling NaN pixels with {NAN_FILL_VALUE}."
                        )
                        warnings.warn(warning_message) 
                        nan_mask = torch.isnan(processed_image)
                        processed_image[nan_mask] = torch.tensor(NAN_FILL_VALUE, dtype=processed_image.dtype)

                    images[name] = processed_image
            current_poses = {name: f[f'/poses_dict/{name}'][step_idx] for name in self.pose_names}
            current_poses_list = [
                f[f'/joints_dict/joints_position_state'][step_idx][:3] if name == 'astribot_chassis' 
                else current_poses[name] 
                for name in self.pose_names
            ]
            current_pose_raw = np.concatenate(current_poses_list).astype(np.float32)

            actions_raw = []
            episode_len = len(f['time'])
            for i in range(self.Ta):
                future_idx = min(step_idx + i, episode_len - 1)
                future_poses_list = [
                    f[f'/joints_dict/joints_position_state'][future_idx][:3] if name == 'astribot_chassis'
                    else f[f'/poses_dict/{name}'][future_idx]
                    for name in self.pose_names
                ]
                action_step = np.concatenate(future_poses_list)
                actions_raw.append(action_step)
            action_sequence_raw = np.array(actions_raw, dtype=np.float32)

        current_pose_normalized = self.normalize_pose(torch.from_numpy(current_pose_raw))
        current_pose_normalized[:3] = 0.0
        action_sequence_normalized = self.normalize_pose(torch.from_numpy(action_sequence_raw))

        return {
            'images': images,
            'pcd_coords': pcd_coords,
            'pcd_feats': pcd_feats,
            'pose': current_pose_normalized,
            'action': action_sequence_normalized
        }