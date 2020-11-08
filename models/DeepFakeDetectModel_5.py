import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
from features.encoders import *
import torch.nn.functional as F


class DeepFakeDetectModel_5(nn.Module):
    """
    max_num_frames are passed as stacked to the model.
    For each frame, features are extracted. Features get applied AdaptiveAvgPool2d, flatten
    and passed as a single feature-map to classifier, expects DFDCDataset.

    Param choices in config.yml:

    cnn_encoder:
      default: ['tf_efficientnet_b0_ns', 'tf_efficientnet_b7_ns'] # choose either one
    training:
      train_size: 1
      valid_size: 1
      test_size: 1
      params:
        model_name: 'DeepFakeDetectModel_5'
        label_smoothing: 0.1 # 0 to disable this, or any value less than 1
        train_transform: ['simple', 'complex'] # choose either of the data augmentation
        batch_format: 'stacked' # Do not change
        # Adjust epochs, learning_rate, batch_size , fp16, opt_level
        epochs: 5
        learning_rate: 0.001
        batch_size: 4
        fp16: True
        opt_level: 'O1'
        dataset: ['optical', 'plain'] # choose either of the data type
        max_num_frames: 5 # Adjust this value.
        random_sorted: [True, False] # if True: frames are selected randomly, else frames are selected from index 0. in
                                     # in either case, frames are sorted.
        fill_empty: False # if True, delta number of empty frames (all 0's) are appended.
                          # delta = max_num_frames - available frames for the sample
                          # if False, min(max_num_frames, total_frames_for_the_samples) frames are returned.
                          # setting False may break the model as it expects at least max_num_frames.
    """

    def __init__(self, frame_dim=None, max_num_frames=5, encoder_name=None):
        super().__init__()
        self.image_dim = frame_dim
        self.max_num_frames = max_num_frames
        self.num_of_classes = 1
        self.encoder = encoder_params[encoder_name]["init_op"]()
        self.encoder_flat_feature_dim = encoder_params[encoder_name]["flat_features_dim"]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        classifier_in = self.encoder_flat_feature_dim * self.max_num_frames
        classifier_hidden = 1024
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
            f = self.avg_pool(f).flatten(1)
            features.append(f)

        features_stack = torch.stack(features)
        features_stack = features_stack.contiguous().view(batch_size, -1)
        out = self.classifier(features_stack)
        return out
