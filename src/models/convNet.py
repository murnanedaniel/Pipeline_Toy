import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, kern_1 = 5, hidden_dim_1 = 6, kern_2 = 5, hidden_dim_2 = 16, hidden_dim_3 = 120, hidden_dim_4 = 84, output_dim = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, hidden_dim_1, kern_1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(hidden_dim_1, hidden_dim_2, kern_2)
        self.flat_num = int(np.ceil( (np.ceil( (32 - kern_1) / 2) - kern_2 ) /2 ))
#         print(self.flat_num)
        self.fc1 = nn.Linear(hidden_dim_2 * self.flat_num * self.flat_num, hidden_dim_3)
        self.fc2 = nn.Linear(hidden_dim_3, hidden_dim_4)
        self.fc3 = nn.Linear(hidden_dim_4, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features