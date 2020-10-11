import torch
import torch.nn as nn
from features.encoders import *


class DeepFakeDetectModel_2(nn.Module):
    """
    Gets max_num_frames frames for each video as stacked. Extract features from each frame
    using encoder and pass these features to LSTM. Output of LSTM is passed for classification.
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
        self.lstm_dropout = lstm_dropout

        self.rnn = nn.LSTM(input_size=self.lstm_in_dim, hidden_size=self.lstm_hidden_dim,
                           num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional,
                           dropout=self.lstm_dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim * self.max_num_frames, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        features = []
        # extract features of each frame and save them in features list, process batch-wise
        for b in range(batch_size):
            f = self.encoder.forward_features(x[b])
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
