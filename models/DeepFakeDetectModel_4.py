import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
from features.encoders import *
import torch.nn.functional as F


class DeepFakeDetectModel_4(nn.Module):
    """
    Use one frame from each video for classification, expects DFDCDataset class for dataset
    Same as DeepFakeDetectModel_3 but this one uses either of the encoder from params, also
    uses AdaptiveAvgPool2d.

    Param choices in config.yml:

    cnn_encoder:
      default: ['tf_efficientnet_b0_ns', 'tf_efficientnet_b7_ns'] # choose either one
    training:
      train_size: 1
      valid_size: 1
      test_size: 1
      params:
        model_name: 'DeepFakeDetectModel_4'
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
        random_sorted: [True, False] # when True: a random frame is selected, when False: frame num#0 is selected.
        fill_empty: False # Do not change. Not used
        max_num_frames: 1 # Do not change. Not used

    """

    def __init__(self, frame_dim=None, encoder_name=None):
        super().__init__()
        self.image_dim = frame_dim
        self.num_of_classes = 1
        self.encoder = encoder_params[encoder_name]["init_op"]()
        self.encoder_flat_feature_dim = encoder_params[encoder_name]["flat_features_dim"]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

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
        if num_frames != 1:
            raise Exception("Number of frames expected one only in DeepFakeDetectModel_3")
        x = torch.squeeze(x, dim=1)
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.classifier(x)
        return x
