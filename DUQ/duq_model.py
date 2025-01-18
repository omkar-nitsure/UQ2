import torch
import torch.nn as nn
from torch.nn import functional as F

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def project(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, input):

        x = self.project(input)
        x = self.fc3(x)
        return x


class CNN_DUQ(LeNet):
    def __init__(
        self,
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 84), 0.05)
        )

        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

    def update_embeddings(self, x, y):
        z = self.last_layer(self.project(x))
        
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

    def last_layer(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    def output_layer(self, z):
        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        distances = (diff**2).mean(1).div(2 * self.sigma**2)
        exp_dists = (-(diff**2)).mean(1).div(2 * self.sigma**2).exp()

        return distances, exp_dists

    def forward(self, x):
        z = self.last_layer(self.project(x))
        dists, y_pred = self.output_layer(z)

        return y_pred, dists