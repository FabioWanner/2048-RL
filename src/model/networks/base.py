from abc import ABCMeta, abstractmethod
from typing import List

import torch
from torch import nn

from src.engine.typings import State


class Network(nn.Module, metaclass=ABCMeta):

    def __init__(
        self,
        size: int,
        n_actions: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.size = size
        self.n_actions = n_actions
        self.device = device

    @abstractmethod
    def forward(self, states: List[State]):
        pass
