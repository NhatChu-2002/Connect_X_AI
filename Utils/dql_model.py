import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class DQNCNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNCNN, self).__init__()
        c, h, w = input_shape 
        
        self.conv1 = nn.Conv2d(c, 64, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(self._get_conv_output_size(c, h, w), 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.dropout = nn.Dropout(0.25)


    def _get_conv_output_size(self, c, h, w):
        x = torch.zeros(1, c, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))     
        x = F.relu(self.conv2(x))      
        x = x.view(x.size(0), -1)      
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)    