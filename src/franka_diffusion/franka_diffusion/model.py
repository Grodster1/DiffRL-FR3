import math
import torch 
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be an even number"
        self.dim = dim
        
    def forward(self, timesteps: torch.Tensor):
        device = timesteps.device
        half_dim = self.dim // 2
        
        embedding = math.log(10000.0) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embedding)
        
        # Broadcasting of time and freq
        # timesteps[:, None]: [batch] -> [batch, 1]
        # embedding[None, :] [half_dim] -> [1, half_dim]
        embedding = timesteps.float()[:, None] * embedding[None, :]  # [batch, half_dim]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)         
        
        return embedding
        
        
class Conv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 5, n_groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding = kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )
        
    def forward(self, x):
        return self.block(x)
    
    
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
