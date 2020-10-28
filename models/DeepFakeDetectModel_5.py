import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
from features.encoders import *
import torch.nn.functional as F


class DeepFakeDetectModel_5(nn.Module):
    """
    max_num_frames are passed stacked.
    For each frame, features are extracted. Features are flatten and passed as a
    single feature-map to classifier.

    Expects original method of training/testing with DFDCDataset class

    sample valid params for config.yml
    params:
        model_name: 'DeepFakeDetectModel_5'
        max_num_frames: 5
        batch_size: 32
        random_sorted: True
        expand_label_dim: False
        epochs: 50
        learning_rate: 0.001
        fill_empty: True

    """
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
            nn.Linear(classifier_hidden, self.num_of_classes)
        )

    def forward(self, x):
        # x.shape = batch_size x max_num_frames x color_channel x image_h x image_w
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        features = []
        # extract features of each frame and save them in features list, process batch-wise
        for b in range(batch_size):
            # x[b].shape = max_num_frames x color_channel x image_h x image_w
            # in passing x[b] to encoder.forward_features(),  max_num_frames acts as batch_size for the video
            # as passing number of frames in batches.
            f = self.encoder.forward_features(x[b])
            f = f.flatten()
            features.append(f)

        features_stack = torch.stack(features)
        features_stack = features_stack.contiguous().view(batch_size, -1)
        out = self.classifier(features_stack)
        return out
