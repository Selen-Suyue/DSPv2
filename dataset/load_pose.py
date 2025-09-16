import json
from pathlib import Path
import numpy as np
def load_pose_limits(task_name="default"):
    config_path = Path(__file__).parent / "pose.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if task_name not in config:
            print(f"Warning: Task '{task_name}' not found in pose limits config, using default.")
            task_name = "default"
        min_pose = np.array(config[task_name]["MIN_POSE"], dtype=np.float32)
        max_pose = np.array(config[task_name]["MAX_POSE"], dtype=np.float32)
        print(f'Pose Config name is {task_name}')
        return min_pose, max_pose
        
    except Exception as e:
        print(f"Error loading pose limits config: {e}")
        return (
            np.array([-0.013, -0.005, -0.002, -0.0, -0.0, 0.8, -0.2, 0.0, -0.3, 0.8, 0.2, 0.1, 0.2, -0.0, 0.0, 0.3, 0.6, -0.2, 0.2, -0.4, 0.2, 0.0, -0.0, 0.5, 0.4, -0.0, -0.0, -0.0, 0.8, -0.7, 0.2, 0.2, 0.5]),
            np.array([0.011, 0.005, 0.002, 0.2, 1e-5, 1.2, 0.1, 0.5, 0.2, 1.0, 0.5, 0.3, 1.1, 0.4, 0.3, 0.7, 0.9, 91.7, 0.5, -0.1, 1.1, 0.3, 0.4, 0.8, 0.8, 89.8, 0.3, 1e-5, 1.3, -0.5, 0.6, 0.5, 0.7])
        )