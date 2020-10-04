import torch
import random
import torch.nn as nn
import sys
from utils import *
from data_utils.utils import *
from models.DeepFakeDetectModel_1 import *
from data_utils.datasets import DFDCDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import multiprocessing
import numpy as np
import torchvision
from features.encoders import *
from torchvision.transforms import transforms
import torch.nn.functional as F


class DeepFakeDetectModel_2(nn.Module):
    def __init__(self, frame_dim=None, encoder_name=None, max_num_frames=None, lstm_hidden_dim=1024, lstm_num_layers=1,
                 lstm_bidirectional=False, lstm_dropout=0):
        super().__init__()
        self.frame_dim = frame_dim
        self.max_num_frames = max_num_frames
        self.encoder = encoder_params[encoder_name]["init_op"]()
        self.encoder_flat_feature_dim = encoder_params[encoder_name]["flat_features_dim"]
        # self.lstm_in_dim = self.max_num_frames * self.encoder_flat_feature_dim
        self.lstm_in_dim = self.encoder_flat_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_dropout = lstm_dropout

        # print(f'creating LSTM')
        # print(f'self.lstm_in_dim = {self.lstm_in_dim}')
        self.rnn = nn.LSTM(input_size=self.lstm_in_dim, hidden_size=self.lstm_hidden_dim,
                           num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional,
                           dropout=self.lstm_dropout, batch_first=True)
        # print(f'Done creating LSTM')
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim * self.max_num_frames, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        features = []
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

    """
    def init_lstm_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).zero_().cuda(),
                  weight.new(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).zero_().cuda())

        return hidden
    """

    def init_lstm_hidden(self, batch_size):
        hidden = (torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).cuda(),
                  torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).cuda())

        return hidden
