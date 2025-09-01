import math

import torch
from torch import nn

from src.model.networks.base import Network


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding="same")
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding="same")
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding="same")
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding="same")

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)


class DQNConvolutional(Network):
    def __init__(
        self,
        size,
        n_actions,
        device: torch.device | None = None,
    ):
        super().__init__(size, n_actions, device)
        self.conv1 = ConvBlock(self.size * self.size, 2048)
        self.conv2 = ConvBlock(2048, 2048)
        self.conv3 = ConvBlock(2048, 2048)
        self.dense1 = nn.Linear(2048 * size * size, 1024)
        self.dense6 = nn.Linear(1024, n_actions)

    def forward(self, states):
        flattened = [x for state in states for row in state for x in row]
        as_log = [0 if e == 0 else int(math.log(e, 2)) for e in flattened]
        as_log_tensor = torch.tensor(as_log, dtype=torch.long, device=self.device)
        one_hot = (
            nn.functional.one_hot(as_log_tensor, num_classes=self.size * self.size)
            .float()
            .flatten()
        )

        x = one_hot.reshape(
            len(states), self.size, self.size, self.size * self.size
        ).permute(0, 3, 1, 2)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = nn.functional.dropout(self.dense1(x))
        x = self.dense6(x)

        assert x.shape == (len(states), self.n_actions)
        return x
