import torch
import random
import torch.nn as nn


class DeepFakeDetectModel_1(nn.Module):
    def __init__(self, frame_dim=None):
        super().__init__()
        self.frame_dim = frame_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.frame_dim * self.frame_dim * 3, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        x1 = x.reshape(batch_size, num_frames, -1)
        return self.classifier(x1[:, 0, :])
