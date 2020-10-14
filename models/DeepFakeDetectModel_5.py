import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
from features.encoders import *
import torch.nn.functional as F


class DeepFakeDetectModel_5(nn.Module):
    def __init__(self, frame_dim=None, max_num_frames=5, encoder_name=None):
        super().__init__()
        self.image_dim = frame_dim
        self.max_num_frames = max_num_frames
        self.num_of_classes = 2
        self.encoder = encoder_params[encoder_name]["init_op"]()
        self.encoder_flat_feature_dim = encoder_params[encoder_name]["flat_features_dim"]
        classifier_in = self.encoder_flat_feature_dim * self.max_num_frames
        classifier_hidden = 1024 #int(classifier_in * 0.05)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, classifier_hidden),
            nn.Dropout(0.50),
            nn.ReLU(),
            nn.Linear(classifier_hidden, 2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        features = []
        # extract features of each frame and save them in features list, process batch-wise
        for b in range(batch_size):
            f = self.encoder.forward_features(x[b])
            f = f.flatten()
            features.append(f)

        features_stack = torch.stack(features)
        features_stack = features_stack.contiguous().view(batch_size, -1)
        out = self.classifier(features_stack)
        return out
