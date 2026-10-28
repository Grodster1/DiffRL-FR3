import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from .model import ConditionalUnet1D


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim = 17, action_dims = 4, horizon = 16, n_obs_steps = 2,
                 n_action_steps = 8, num_train_steps = 100, num_inference_steps = 10):
        
        super().__init__()
        assert horizon % 4 == 0, "Horizon must be divisible by 4"

        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps

        self.model = ConditionalUnet1D(input_dim = action_dims,
                                       global_cond_dim = n_obs_steps * obs_dim,
                                       cond_predict_scale = True)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps = num_train_steps,
                                             beta_schedule = "squaredcos_cap_v2",
                                             clip_sample = True, prediction_type = "epsilon")
        self.inference_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)

    def compute_loss(self, batch):
        obs, action = batch["obs"], batch["action"]
        B = action.shape[0]

        cond = obs.reshape(B, -1)                       # (B, 34)

        noise = torch.randn_like(action)
        k = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                          (B,), device = action.device)

        noisy_action = self.noise_scheduler.add_noise(action, noise, k)
        pred_noise = self.model(noisy_action, k, global_cond = cond)

        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def predict_action(self, obs):
        # obs: (B, n_obs_steps, obs_dim)
        B = obs.shape[0]
        cond = obs.reshape(B, -1)
        
        x = torch.randn(B, self.horizon, self.action_dims, device = obs.device)
        
        self.inference_scheduler.set_timesteps(self.num_inference_steps)
        for k in self.inference_scheduler.timesteps:
            pred = self.model(x, k, global_cond = cond)
            x = self.inference_scheduler.step(pred, k, x).prev_sample
        
        x = x.clamp(-1.0, 1.0)
        start = self.n_obs_steps - 1
        return x[:, start : start + self.n_action_steps]