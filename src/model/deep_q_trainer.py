import math
import random
from pathlib import Path
from typing import Callable, List
from uuid import uuid4

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from src.engine.engine import Direction, Engine2048
from src.engine.typings import State
from src.model.networks.tutorial_network import DQNTutorial
from src.model.parameters import Parameters
from src.model.progress_tracker import ProgressTracker
from src.model.replay_memory import ReplayMemory, MemoryFragment


class DeepQTrainer:
    optimizer: Optimizer
    action_to_direction = {
        0: Direction.UP,
        1: Direction.DOWN,
        2: Direction.LEFT,
        3: Direction.RIGHT,
    }
    training_id = str(uuid4())
    episode = 0

    def __init__(
        self,
        board_size: int = 4,
        parameters: Parameters = Parameters(),
        optimizer_factory: Callable[
            [ParamsT, float | Tensor], Optimizer
        ] = torch.optim.Adam,
        engine_factory: Callable[[int, int | None], Engine2048] = Engine2048,
        network_factory: Callable[[int, int], torch.nn.Module] = DQNTutorial,
        memory_factory: Callable[[int], ReplayMemory] = ReplayMemory,
        device: torch.device | None = None,
        seed: int | None = None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.parameters = parameters

        self.engine = engine_factory(board_size, seed)
        self.num_states = board_size * board_size
        self.num_actions = len(self.action_to_direction)

        self.device = device
        self.policy_network = network_factory(self.num_states, self.num_actions).to(
            device
        )
        self.target_network = network_factory(self.num_states, self.num_actions).to(
            device
        )

        self.memory = memory_factory(parameters.replay_memory_size)
        self._optimizer_factory = optimizer_factory
        self._loss_fn = nn.MSELoss()

    def _initialize_networks(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.policy_network.train()
        self.optimizer = self._optimizer_factory(
            self.policy_network.parameters(), self.parameters.learning_rate
        )

    @property
    def epsilon(self):
        return self.parameters.epsilon_min + (
            self.parameters.epsilon_max - self.parameters.epsilon_min
        ) * math.exp(-1.0 * self.episode / self.parameters.epsilon_decay_rate)

    @torch.no_grad()
    def _select_action(self, state: State):
        if self.rng.random() > self.epsilon:
            return (
                self.policy_network(self.state_to_network_input(state)).argmax().item()
            )
        else:
            return self.rng.choice(range(0, self.num_actions))

    @staticmethod
    def state_to_network_input(state: State) -> torch.Tensor:
        return torch.tensor([x for row in state for x in row], dtype=torch.float)

    @torch.no_grad()
    def optimize(self, memory, policy_dqn, target_dqn, optimizer):
        pass

    @staticmethod
    def run_episode(
        engine: Engine2048,
        select_fn: Callable[[State], int],
        action_to_direction: dict[int, Direction],
    ) -> List[MemoryFragment]:
        moves = []
        while not engine.game_over:
            previous_state = engine.state
            previous_score = engine.score

            action = select_fn(engine.state)
            engine.evolve(action_to_direction[action])

            state = engine.state
            score = engine.score - previous_score

            moves.append(MemoryFragment(previous_state, action, state, score))
        return moves

    def train(self, episodes: int, out: Path, description: str | None = None):
        self.training_id = str(uuid4())
        data_dir = out / self.training_id
        data_dir.mkdir(parents=True)
        print(f"Training with id: {self.training_id}, output directory: {data_dir}")

        tracker = ProgressTracker(data_dir, to_console=True)
        tracker.write_metadata(
            dict(
                id=self.training_id,
                optimizer_factory=self._optimizer_factory.__name__,
                parameters=self.parameters.__dict__,
                seed=self.seed,
                description=description,
            )
        )

        self._initialize_networks()

        for self.episode in range(episodes):
            self.engine.reset()

            moves = self.run_episode(
                self.engine, self._select_action, self.action_to_direction
            )

            tracker.track(
                episode=self.episode,
                score=self.engine.score,
                number_of_moves=len(moves),
                max_tile=max(max(moves[-1].state)),
                epsilon=self.epsilon,
            )

            self.optimize(
                self.memory, self.policy_network, self.target_network, self.optimizer
            )
