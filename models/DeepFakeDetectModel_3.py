import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
from features.encoders import *
import torch.nn.functional as F


class DeepFakeDetectModel_3(nn.Module):
    """
    A custom CNN, uses one frame from the video, expects DFDCDataset for dataset

    Param choices in config.yml:

    cnn_encoder:
      default: 'tf_efficientnet_b0_ns' # Do not change. Not used
    training:
      train_size: 1
      valid_size: 1
      test_size: 1
      params:
        model_name: 'DeepFakeDetectModel_3'
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

    def __init__(self, frame_dim=None):
        super().__init__()
        self.image_dim = frame_dim
        self.num_of_classes = 1
        self.padding = 2
        self.conv1_out_channel = 6
        self.conv1_kernel_size = 3
        self.stride = 1
        self.conv2_out_channel = 24
        self.conv2_kernel_size = 4
        self.conv3_out_channel = 32
        self.conv3_kernel_size = 5

        conv1_neurons = int((self.image_dim - self.conv1_kernel_size + 2 * self.padding) / self.stride + 1)
        maxpool2d_1_neurons = int(conv1_neurons / 2)
        conv2_neurons = ((maxpool2d_1_neurons - self.conv2_kernel_size + 2 * self.padding) / self.stride + 1)
        maxpool2d_2_neurons = int(conv2_neurons / 2)
        conv3_neurons = ((maxpool2d_2_neurons - self.conv3_kernel_size + 2 * self.padding) / self.stride + 1)
        maxpool2d_3_neurons = int(conv3_neurons / 2)

        self.linear1_in_features = int(maxpool2d_3_neurons * maxpool2d_3_neurons * self.conv3_out_channel)

        self.linear1_out_features = int(self.linear1_in_features * 0.30)
        self.linear2_out_features = int(self.linear1_out_features * 0.05)

        self.features = nn.Sequential(
            nn.Conv2d(3, self.conv1_out_channel, self.conv1_kernel_size,
                      stride=self.stride, padding=(self.padding, self.padding)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.conv1_out_channel, self.conv2_out_channel, self.conv2_kernel_size,
                      stride=self.stride, padding=(self.padding, self.padding)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.conv2_out_channel, self.conv3_out_channel, self.conv3_kernel_size,
                      stride=self.stride, padding=(self.padding, self.padding)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.linear1_in_features, self.linear1_out_features),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.linear1_out_features, self.linear2_out_features),
            nn.ReLU(),
            nn.Linear(self.linear2_out_features, self.num_of_classes)
        )

    def forward(self, x):
        # x shape = batch_size x num_frames x color_channels x image_h x image_w
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        # there is only one frame expected in this model, so squeeze the num_frames dim
        if num_frames != 1:
            raise Exception("Number of frames expected one only in DeepFakeDetectModel_3")
        x = torch.squeeze(x, dim=1)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
