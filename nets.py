import torch.nn as nn
import torch.nn.functional as F

class TwoLayer(nn.Module):
    def __init__(self, dim):
        super(TwoLayer, self).__init__()
        self.fc1 = nn.Linear(dim, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x