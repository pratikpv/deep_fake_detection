import torch
import torch.nn as nn
from features.encoders import *


class DeepFakeDetectModel_2(nn.Module):
    """
    Gets max_num_frames frames for each video as stacked. Extract features from each frame
    using encoder and pass these features to LSTM. Output of LSTM is passed for classification.
    Output of all LSTM cells are sent to (classification) in an seq-to-seq manner.


    cnn_encoder:
      default: ['tf_efficientnet_b0_ns', 'tf_efficientnet_b7_ns'] # choose either one
    training:
      train_size: 1
      valid_size: 1
      test_size: 1
      params:
        model_name: 'DeepFakeDetectModel_2'
        label_smoothing: 0.1 # 0 to disable this, or any value less than 1
        train_transform: ['simple', 'complex'] # choose either of the data augmentation
        batch_format: 'stacked' # Do not change
        # Adjust epochs, learning_rate, batch_size , fp16, opt_level
        epochs: 5
        learning_rate: 0.001
        batch_size: 4
        fp16: True
        opt_level: 'O0'
        dataset: ['optical', 'plain'] # choose either of the data type
        max_num_frames: 5 # Adjust this value.
        random_sorted: [True, False] # if True: frames are selected randomly, else frames are selected from index 0. in
                                     # in either case, frames are sorted.
        fill_empty: [True, False] # if True, delta number of empty frames (all 0's) are appended.
                                  # delta = max_num_frames - available frames for the sample
                                  # if False, min(max_num_frames, total_frames_for_the_samples) frames are returned.
                                  # setting False may break the model as it expects at least max_num_frames.
        lstm:
          hidden_dim: 1024 # Tune
          num_layers: 1 # Tune
          dropout: 0.30 # Tune, applied only when num_layers more than one.

    """
    def __init__(self, frame_dim=None, encoder_name=None, max_num_frames=None, lstm_hidden_dim=1024, lstm_num_layers=1,
                 lstm_bidirectional=False, lstm_dropout=0):
        super().__init__()
        self.frame_dim = frame_dim
        self.max_num_frames = max_num_frames
        self.encoder = encoder_params[encoder_name]["init_op"]()
        self.encoder_flat_feature_dim = encoder_params[encoder_name]["flat_features_dim"]
        self.lstm_in_dim = self.encoder_flat_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_dropout = 0
        if self.lstm_num_layers > 1:
            self.lstm_dropout = lstm_dropout
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.LSTM(input_size=self.lstm_in_dim, hidden_size=self.lstm_hidden_dim,
                           num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional,
                           dropout=self.lstm_dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim * self.max_num_frames, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        features = []
        # extract features of each frame and save them in features list, process batch-wise
        for b in range(batch_size):
            f = self.encoder.forward_features(x[b])
            f = self.avg_pool(f).flatten(1)
            f = f.view(num_frames, -1)
            features.append(f)

        features = torch.stack(features)

        lstm_hidden = self.init_lstm_hidden(batch_size)
        r_output, lstm_hidden = self.rnn(features, lstm_hidden)
        out = r_output.contiguous().view(batch_size, -1)
        out = self.classifier(out)

        return out

    def init_lstm_hidden(self, batch_size):
        hidden = (torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).cuda(),
                  torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).cuda())

        return hidden
