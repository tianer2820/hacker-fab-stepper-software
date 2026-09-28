from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


class LatentVisionCNN(nn.Module):
    """
    Explicit convolutional neural network for same-size image-to-image mapping:
      - Input:  RGB image with shape (B, 3, H, W)
      - Output: Single-channel heatmap with shape (B, 1, H, W)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters

        f = base_filters  # e.g., 32

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),

            nn.Conv2d(f, f, kernel_size=5, padding=2, stride=2, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(f, f*2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(f*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(f*2, f*2, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(f*2, f*2, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(f*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(f*2, f, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(f, f, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),

            nn.Conv2d(f, f, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(f, out_channels, kernel_size=5, padding=2, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


def load_latent_vision_model(
    checkpoint_path: Union[str, Path], device: str = "cpu"
) -> Optional[LatentVisionCNN]:
    """Attempts to load the LatentVisionCNN model from a checkpoint file.

    Returns the loaded model in eval mode, or None if loading fails.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        print(f"[Latent Vision] Checkpoint file not found: {ckpt_file}")
        return None

    try:
        model = LatentVisionCNN()
        ckpt = torch.load(ckpt_file, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            model = ckpt
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Latent Vision] Failed to load latent vision checkpoint: {e}")
        return None


