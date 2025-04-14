import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class DuelingDQNCNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQNCNN, self).__init__()
        c, h, w = input_shape

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.25)
        )

        conv_out_size = self._get_conv_output_size(c, h, w)

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_actions)
        )

    def _get_conv_output_size(self, c, h, w):
        x = torch.zeros(1, c, h, w)
        x = self.feature_extractor(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        features = self.feature_extractor(x)
        flat = features.view(features.size(0), -1)
        value = self.value_stream(flat)
        advantage = self.advantage_stream(flat)
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals
