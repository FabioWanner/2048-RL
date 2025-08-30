from collections import deque
from dataclasses import dataclass
from random import Random
from typing import List

from src.engine.typings import State


@dataclass
class MemoryFragment:
    state: State
    action: int
    next_state: State
    score: int
    terminal: bool = False


class ReplayMemory:
    def __init__(self, capacity, seed: int | None = None):
        self.rng = Random(seed)
        self.memory = deque([], maxlen=capacity)

    def push(self, fragment: MemoryFragment) -> None:
        self.memory.append(fragment)

    def sample(self, batch_size) -> List[MemoryFragment]:
        return self.rng.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
