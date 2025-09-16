import copy
import os
import cv2
import sys
import argparse
import rospy
from sensor_msgs.msg import CompressedImage
from core.astribot_api.astribot_client import Astribot
from scipy.spatial.transform import Rotation
import pickle
import dataclasses
import pathlib
from PIL import Image
import logging
import numpy as np
import MinkowskiEngine as ME
import matplotlib
matplotlib.use('Agg')
import time
import open3d as o3d
import torchvision.transforms as T
from copy import deepcopy
import torch
from torchvision import transforms

from policy import dspv2
from collections import deque
from dataset.constants import  (
    IMG_MEAN, IMG_STD, MIN_BOX, MAX_BOX,
    NUM_PC_POINTS, INT_JOINTS, T_HEAD_CAM, CAM_INTRINSIC) 
from dataset.load_pose import load_pose_limits


_original_termios_settings = None


class RGBDRead:
    def __init__(self, astribot, policy_type):
        self.astribot = astribot
        self.policy_type = policy_type
        self.timeout = 100
        self.head_rgb, self.head_depth = None, None
        self.torso_image, self.left_image, self.right_image = None, None, None
        self.last_head_rgb_time = 0
        self.last_head_depth_time = 0
        self.last_torso_time = 0
        self.last_left_time = 0
        self.last_right_time = 0

        rospy.Subscriber('/astribot_camera/head_rgbd/color_compress/compressed', CompressedImage, self.head_rgb_callback)
        rospy.Subscriber('/astribot_camera/head_rgbd/depth_compress', CompressedImage, self.head_depth_callback)
        rospy.Subscriber('/astribot_camera/torso_rgbd/color_compress/compressed', CompressedImage, self.torso_callback)
        rospy.Subscriber('/astribot_camera/left_wrist_rgbd/color_compress/compressed', CompressedImage, self.left_callback)
        rospy.Subscriber('/astribot_camera/right_wrist_rgbd/color_compress/compressed', CompressedImage, self.right_callback)
        print(f"\033[94mInfo: Subscribing to torso, left, and right cameras for policy '{self.policy_type}'.\033[0m")

    def head_rgb_callback(self, msg):
        self.head_rgb = self._convert_compressed_rgb(msg)
        self.last_head_rgb_time = time.time()

    def head_depth_callback(self, msg):
        self.head_depth = self._convert_compressed_depth(msg)
        self.last_head_depth_time = time.time()

    def torso_callback(self, msg):
        self.torso_image = self._convert_compressed_rgb(msg)
        self.last_torso_time = time.time()

    def left_callback(self, msg):
        self.left_image = self._convert_compressed_rgb(msg)
        self.last_left_time = time.time()

    def right_callback(self, msg):
        self.right_image = self._convert_compressed_rgb(msg)
        self.last_right_time = time.time()

    def _convert_compressed_rgb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def _convert_compressed_depth(self, msg):
        return np.frombuffer(msg.data, np.uint16).reshape(720, 1280)

    def get_rgbd(self):
        now = time.time()
        time_out_flag = False
        
        if self.head_rgb is None or now - self.last_head_rgb_time > self.timeout:
            print("\033[91mRGBDRead: Timeout or not received, head rgb is None\033[0m")
            time_out_flag = True
        if self.head_depth is None or now - self.last_head_depth_time > self.timeout:
            print("\033[91mRGBDRead: Timeout or not received, head depth is None\033[0m")
            time_out_flag = True
        if self.left_image is None or now - self.last_left_time > self.timeout:
            print("\033[91mRGBDRead: Timeout or not received, left_image is None\033[0m")
            time_out_flag = True
        if self.right_image is None or now - self.last_right_time > self.timeout:
            print("\033[91mRGBDRead: Timeout or not received, right_image is None\033[0m")
            time_out_flag = True
        if self.torso_image is None or now - self.last_torso_time > self.timeout:
            print("\033[91mRGBDRead: Timeout or not received, torso_image is None\033[0m")
            time_out_flag = True
        
        if time_out_flag:
            return None
            
        rgbd_dict = {
            "rgb": self.head_rgb.copy(),
            "depth": self.head_depth.copy(),
        }
        rgbd_dict.update({
            "torso": self.torso_image.copy(),
            "left": self.left_image.copy(),
            "right": self.right_image.copy(),
        })
        return rgbd_dict

    def get_rgb_obs_dict(self, astribot_names):
        poses = np.concatenate(self.astribot.get_current_cartesian_pose(astribot_names))
        chassis = np.array(self.astribot.get_current_joints_position(['astribot_chassis'])).flatten()
        pose_state = np.concatenate((chassis,poses))

        pose_world_chassis = self.astribot.get_current_cartesian_pose([self.astribot.chassis_name])[0]
        pose_world_head = self.astribot.get_current_cartesian_pose([self.astribot.head_name])[0]
        
        rgbd_dict = self.get_rgbd()
        if rgbd_dict is None:
            return None

        obs_dict = {
            "pose_state": pose_state,
            "pose_world_head": pose_world_head,
            "pose_world_chassis": pose_world_chassis
        }
        obs_dict.update(rgbd_dict)
        return obs_dict


class BaseDataProcessor:
    def __init__(self, args):
        self.T_head_cam = np.array(args.T_head_cam)
        self.cam_intrinsic = np.array(args.cam_intrinsic)
        self.aug = T.Compose([
            T.RandomResizedCrop((224, 224), scale=(1.0, 1.0), ratio=(16/9, 16/9)),
        ])
        self.im_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

    def _pose_to_transformation_matrix(self, pose):
        tx, ty, tz, qx, qy, qz, qw = pose
        rotation_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = np.array([tx, ty, tz])
        return transformation_matrix

    def _rgbd_to_pcd_in_cam(self, rgb_image, depth_image, depth_scale=1000.0, depth_max=3.0):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=1280, height=720,
            fx=self.cam_intrinsic[0, 0], fy=self.cam_intrinsic[1, 1],
            cx=self.cam_intrinsic[0, 2], cy=self.cam_intrinsic[1, 2]
        )
        o3d_rgb = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        o3d_depth = o3d.geometry.Image(depth_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb, o3d_depth, depth_scale=depth_scale, depth_trunc=depth_max, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        return pcd

    def _transform_and_crop_pcd(self, pcd, pose_world_head, pose_world_chassis):
        T_world_head = self._pose_to_transformation_matrix(pose_world_head)
        T_world_chassis = self._pose_to_transformation_matrix(pose_world_chassis)
        T_chassis_world = np.linalg.inv(T_world_chassis)
        
        pcd.transform(self.T_head_cam)
        # Transform pcd from head frame to chassis frame
        pcd.transform(T_chassis_world @ T_world_head)

        return pcd.crop(o3d.geometry.AxisAlignedBoundingBox(MIN_BOX, MAX_BOX))
    
    def _process_images(self, obs_rgbd_dict):
        return torch.stack([
            self.im_trans(self.aug(Image.fromarray(cv2.cvtColor(obs_rgbd_dict['rgb'], cv2.COLOR_BGR2RGB)))).float(),
            self.im_trans(self.aug(Image.fromarray(cv2.cvtColor(obs_rgbd_dict['left'], cv2.COLOR_BGR2RGB)))).float(),
            self.im_trans(self.aug(Image.fromarray(cv2.cvtColor(obs_rgbd_dict['right'], cv2.COLOR_BGR2RGB)))).float(),
            self.im_trans(self.aug(Image.fromarray(cv2.cvtColor(obs_rgbd_dict['torso'], cv2.COLOR_BGR2RGB)))).float()
        ])
    
    def process(self, obs_rgbd_dict):
        raise NotImplementedError

class DSPDataProcessor(BaseDataProcessor):
    def __init__(self, args):
        super().__init__(args)

    def _pcd_to_sparse_tensor_inputs(self, pcd):
        points = np.asarray(pcd.points, dtype=np.float32)
        feats = points.copy()
        coords = np.floor(points / 0.005).astype(np.int32)
        return [feats], [coords]
    
    def process(self, obs_rgbd_dict):
        pcd_cam = self._rgbd_to_pcd_in_cam(obs_rgbd_dict['rgb'], obs_rgbd_dict['depth'])
        pcd_cropped = self._transform_and_crop_pcd(pcd_cam, obs_rgbd_dict['pose_world_head'], obs_rgbd_dict['pose_world_chassis'])
        f_lst, c_lst = self._pcd_to_sparse_tensor_inputs(pcd_cropped)
        coords_batch, feats_batch = ME.utils.sparse_collate(c_lst, f_lst)
        
        imgs = self._process_images(obs_rgbd_dict)
        
        return {
            "coords": coords_batch,
            "feats": feats_batch,
            "agent_pos": obs_rgbd_dict['pose_state'],
            "imgs": imgs,
        }



def get_args():
    parser = argparse.ArgumentParser()
    # Add new policy choices
    parser.add_argument('--policy', type=str, required=True, choices=['dspv2'], help='Policy to use for evaluation.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the policy checkpoint.')
    parser.add_argument('--move', action='store_true', help='Actually move the robot.')
    parser.add_argument('--duration', type=float, default=0.5, help='Duration for each move command.')
    parser.add_argument('--task', type=str, default='default', help='Task name to load specific pose limits.')

    args = parser.parse_args()

    args.T_head_cam = T_HEAD_CAM
    args.cam_intrinsic = CAM_INTRINSIC
    return args

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHASSIS_OFFSET_FILE = 'chassis_offset.pkl'
    chassis_offset = None

    astribot = Astribot(high_control_rights=True)
    rgbd_interface = RGBDRead(astribot, args.policy)
    
    print("Waiting for sensor data...")
    while rgbd_interface.get_rgbd() is None:
        time.sleep(0.5)
        print("Still waiting...")
    print("Sensor data received.")

    if os.path.exists(CHASSIS_OFFSET_FILE):
        try:
            with open(CHASSIS_OFFSET_FILE, 'rb') as f:
                chassis_offset = pickle.load(f)
            print(f"Loaded chassis offset from {CHASSIS_OFFSET_FILE}: {chassis_offset}")
            if not isinstance(chassis_offset, np.ndarray) or chassis_offset.shape != (3,):
                raise ValueError("Loaded chassis offset has incorrect format. Reinitializing.")
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            print(f"Error loading chassis offset file: {e}. Reinitializing offset.")
            chassis_offset = None 
    
    if chassis_offset is None:
        print("Chassis offset not found or invalid. Reading current position and saving it.")
        current_chassis_pos = np.array(astribot.get_current_joints_position(['astribot_chassis'])).flatten()
        chassis_offset = current_chassis_pos

        try:
            with open(CHASSIS_OFFSET_FILE, 'wb') as f:
                pickle.dump(chassis_offset, f)
            print(f"Saved current chassis position as offset to {CHASSIS_OFFSET_FILE}: {chassis_offset}")
        except Exception as e:
            print(f"\033[91mError saving chassis offset: {e}. Continuing without saving.\033[0m")
        

        policy = dspv2(Tp=16, Ta=16, input_dim=3, action_dim=33).to(device)
        data_processor = DSPDataProcessor(args)
        

    policy.load_state_dict(torch.load(args.ckpt_path, map_location=device), strict=False)
    policy.eval()
    print(f"Loaded checkpoint for policy '{args.policy.upper()}' from {args.ckpt_path}")
    MIN_POSE, MAX_POSE = load_pose_limits(args.task)
    norm_min = torch.from_numpy(MIN_POSE.astype(np.float32)).to(device)
    norm_max = torch.from_numpy(MAX_POSE.astype(np.float32)).to(device)
    norm_range = norm_max - norm_min
    norm_range[norm_range == 0] = 1e-8

    
    
    joint_action_init = INT_JOINTS
    joint_names = astribot.whole_body_names[1:]
    joint_action = [joint_action_init[:4], joint_action_init[4:11], joint_action_init[11:12],
                    joint_action_init[12:19], joint_action_init[19:20], joint_action_init[20:22]]
    if args.move:
        print("Moving chassis to offset position...")
        astribot.move_joints_position(['astribot_chassis'], [chassis_offset], duration=4.0, use_wbc=True)
        time.sleep(4.0)
        print("Moving to initial joint positions...")
        astribot.move_joints_position(joint_names, joint_action, duration=4.0, use_wbc=False)
    
    time.sleep(4.0)
    print("Initial position reached.")

    astribot_names = ['astribot_torso', 'astribot_arm_left', 'astribot_gripper_left', 'astribot_arm_right', 'astribot_gripper_right', 'astribot_head']
    obs_queue = deque(maxlen=1)
    
    while not obs_queue:
        initial_obs = rgbd_interface.get_rgb_obs_dict(astribot_names)
        if initial_obs:
            obs_queue.append(initial_obs)
        else:
            print("Failed to get initial observation, retrying...")
            time.sleep(1)

    rate = rospy.Rate(1 / args.duration)
    loop_count = 0
    while not rospy.is_shutdown() and loop_count < 500:

        obs_rgbd_dict = obs_queue[0]
        
        obs_dict = data_processor.process(obs_rgbd_dict)
        current_agent_pos_for_policy = obs_dict["agent_pos"].copy()
        move_base = current_agent_pos_for_policy[:3]
        pose_tensor = torch.from_numpy(current_agent_pos_for_policy).unsqueeze(0).to(device)
        pose_data = ((pose_tensor - norm_min) / norm_range).float()
        pose_data[:, :3] = 0.0
        infer_start_time = time.time()
        with torch.no_grad():
            coords, feats = obs_dict["coords"].to(device), obs_dict["feats"].to(device)
            cloud_data = ME.SparseTensor(feats, coords)
            imgs = obs_dict["imgs"].unsqueeze(0).to(device)
            actions = policy(cloud=cloud_data, actions=None, qpos=pose_data, imgs=imgs, batch_size=1)

        actions = actions.squeeze(0).cpu()
        actions = actions * norm_range.cpu() + norm_min.cpu()
        actions = actions.numpy()

        print(f"Loop {loop_count}, Policy: {args.policy.upper()}, Inference time: {time.time() - infer_start_time:.4f}s")

        for loop_idx in range(actions.shape[0]):
            if rospy.is_shutdown(): break
            action = actions[loop_idx]
            astribot_actions = [action[3:10], action[10:17], 1.2*action[17:18], action[18:25], 1.2*action[25:26], action[26:33]]
            move_base += action[:3]
            if args.move:
                astribot.move_joints_position(['astribot_chassis'], [move_base] ,duration=args.duration, use_wbc=True)
                astribot.move_cartesian_pose(astribot_names, 
                                             astribot_actions, duration=args.duration, use_wbc=True)
            if loop_idx == actions.shape[0] - 1:
                new_obs = rgbd_interface.get_rgb_obs_dict(astribot_names)
                if new_obs:
                    obs_queue.append(new_obs)
                else:
                    print("\033[91mWarning: Failed to get new observation, reusing the old one.\033[0m")
            
            rate.sleep()
            
        loop_count += 1

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        main()
    except rospy.ROSInterruptException:
        pass


# python eval.py --move --duration 0.13 --policy dspv2 --ckpt_path logs/dspv2/wine/policy_epoch_150_seed_233.ckpt --task wine
