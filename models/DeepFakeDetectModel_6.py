import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
from features.encoders import *
import torch.nn.functional as F


class DeepFakeDetectModel_6(nn.Module):
    """
    training:
      train_size: 1
      valid_size: 1
      test_size: 1
      checkpoint_path: 'checkpoints'
      params:
        model_name: 'DeepFakeDetectModel_6'
        epochs: 5
        learning_rate: 0.001
        batch_size: 128
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
        # x shape = batch_size x color_channels x image_h x image_w
        batch_size = x.shape[0]
        x = self.encoder.forward_features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
