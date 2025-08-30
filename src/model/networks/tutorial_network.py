import torch
from torch import nn

from src.model.networks.base import Network


class DQNTutorial(Network):
    def __init__(
        self,
        size,
        n_actions,
        device: torch.device | None = None,
    ):
        super().__init__(size, n_actions, device)
        self.layer1 = nn.Linear(size * size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, states):
        flattened = [[x for row in state for x in row] for state in states]
        x = torch.tensor(flattened, dtype=torch.float, device=self.device)
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        assert x.shape == (len(states), self.n_actions)
        return x
