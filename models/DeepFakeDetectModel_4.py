import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
from features.encoders import *
import torch.nn.functional as F


class DeepFakeDetectModel_4(nn.Module):
    """
    Use one frame from each video for classification
    Expects original method of training/testing with DFDCDataset class
    """
    def __init__(self, frame_dim=None, encoder_name=None):
        super().__init__()
        self.image_dim = frame_dim
        self.num_of_classes = 1
        self.encoder = encoder_params[encoder_name]["init_op"]()
        self.encoder_flat_feature_dim = encoder_params[encoder_name]["flat_features_dim"]

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_flat_feature_dim, int(self.encoder_flat_feature_dim * .10)),
            nn.Dropout(0.50),
            nn.ReLU(),
            nn.Linear(int(self.encoder_flat_feature_dim * .10), self.num_of_classes),
        )

    def forward(self, x):
        # x shape = batch_size x num_frames x color_channels x image_h x image_w
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        # there is only one frame expected in this model, so squeeze the num_frames dim
        x = torch.squeeze(x, dim=1)
        x = self.encoder.forward_features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
