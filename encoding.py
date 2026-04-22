import torch
from config import Config

def rate_encoding(x):
    """
    Input: (B, time, feature)
    Output: (T, B, time, feature)

    Normalize per-sample về [0, 1] rồi dùng làm spike probability.
    """
    B = x.shape[0]
    x_flat = x.view(B, -1)
    x_min = x_flat.min(dim=1)[0].view(B, 1, 1)
    x_max = x_flat.max(dim=1)[0].view(B, 1, 1)
    x = (x - x_min) / (x_max - x_min + 1e-9)  # [0, 1]

    spikes = torch.rand(
        (Config.TIME_STEPS, *x.shape),
        device=x.device
    ) < x

    return spikes.float()