import torch
import numpy as np
from pathlib import Path

class DemonstrationDataset(torch.utils.data.Dataset):
    def __init__(self, demo_dirs, horizon=16, n_obs_steps=2, n_action_steps=8, episode_ids=None):
        self.file_paths = []
        if isinstance(demo_dirs, (str, Path)):
            demo_dirs = [demo_dirs]
            
        for directory in demo_dirs:
            dir_path = Path(directory)
            if dir_path.exists() and dir_path.is_dir():
                matched_files = sorted(dir_path.glob("ep_*.npz"))
                self.file_paths.extend(matched_files)
            else:
                raise FileNotFoundError(f"Error: Directory {directory} does not exist or is not a directory")
                
        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .npz files have been found in location {demo_dirs}")
        
        if episode_ids is not None:
            self.file_paths = [self.file_paths[i] for i in episode_ids]
            
        assert horizon >= n_obs_steps - 1 + n_action_steps, "Horizon must cover the obs history and execucet actions"
        
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        
        self.episodes = []
        for path in self.file_paths:
            with np.load(path) as data:
                obs = data["obs"][:-1]      #The final obs has no action connected with it
                action = data["action"]
            assert len(obs) == len(action), f"{path}: obs/action length mismatch"
            self.episodes.append((obs, action))
        
        self.indices = []
        for ep_idx, (_, action) in enumerate(self.episodes):
            T = len(action)
            first = -(n_obs_steps - 1)
            last = T - (n_obs_steps - 1) - n_action_steps
            for s in range(first, last + 1):
                self.indices.append((ep_idx, s))
        
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ep_idx, s = self.indices[idx]
        obs, action = self.episodes[ep_idx]
        T = len(action)
        
        t = np.arange(s, s+self.horizon)
        t_clipped = np.clip(t, 0, T - 1)
        
        obs_window = obs[t_clipped[:self.n_obs_steps]]
        action_window = action[t_clipped]
        action_window[t >= T, :3] = 0.0
        
        return {
            "obs": torch.from_numpy(obs_window),
            "action": torch.from_numpy(action_window)
        }
        
        