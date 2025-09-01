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
from src.model.networks.base import Network
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
        network_factory: Callable[
            [int, int, torch.device | None], Network
        ] = DQNTutorial,
        memory_factory: Callable[[int], ReplayMemory] = ReplayMemory,
        device: torch.device | None = None,
        seed: int | None = None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.parameters = parameters

        self.engine = engine_factory(board_size, seed)
        self.board_size = board_size
        self.num_actions = len(self.action_to_direction)

        self.device = device
        self._network_factory = network_factory
        self.policy_network = network_factory(
            self.board_size, self.num_actions, self.device
        ).to(self.device)
        self.target_network = network_factory(
            self.board_size, self.num_actions, self.device
        ).to(self.device)

        self.memory = memory_factory(parameters.replay_memory_size)
        self._optimizer_factory = optimizer_factory
        self._loss_fn = nn.SmoothL1Loss()

    def _initialize_networks(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.policy_network.train()
        self.optimizer = self._optimizer_factory(
            self.policy_network.parameters(), self.parameters.learning_rate
        )

    def _update_target_network(self):
        if self.episode % self.parameters.target_update_rate == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.policy_network.train()

    def _snapshot_policy_network(self, snapshot_dir: Path):
        snapshot_path = snapshot_dir / "model.pt"
        torch.save(self.policy_network.state_dict(), snapshot_path)

    @property
    def epsilon(self):
        return self.parameters.epsilon_min + (
            self.parameters.epsilon_max - self.parameters.epsilon_min
        ) * math.exp(-1.0 * self.episode / self.parameters.epsilon_decay_rate)

    @torch.no_grad()
    def _select_action(self, state: State):
        if self.rng.random() > self.epsilon:
            return self.policy_network([state]).argmax().item()
        else:
            return self.rng.choice(range(0, self.num_actions))

    @staticmethod
    def optimize(
        parameters: Parameters,
        memory: ReplayMemory,
        policy_network: Network,
        target_network: Network,
        optimizer: Optimizer,
        loss_fn,
        device: torch.device | None = None,
    ):
        if len(memory) < parameters.batch_size:
            return

        batch = memory.sample(parameters.batch_size)

        states = []
        actions = []
        scores = []
        non_terminal_next_states = []
        is_not_terminal = []

        for fragment in batch:
            states.append(fragment.state)
            actions.append(fragment.action)
            scores.append(fragment.score)
            if not fragment.terminal:
                non_terminal_next_states.append(fragment.next_state)
            is_not_terminal.append(not fragment.terminal)

        non_terminal_mask = torch.tensor(
            is_not_terminal, device=device, dtype=torch.bool
        )

        state_action_values = policy_network(states).gather(
            1,
            torch.tensor(actions, device=device, dtype=torch.int).reshape(
                parameters.batch_size, 1
            ),
        )

        next_state_values = torch.zeros(parameters.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_terminal_mask] = (
                target_network(non_terminal_next_states).max(1)[0].detach()
            )
        expected_state_action_values = (
            next_state_values * parameters.discount_factor
        ) + torch.tensor(scores, device=device, dtype=torch.int)

        loss = loss_fn(
            state_action_values,
            expected_state_action_values.reshape(parameters.batch_size, 1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

            if engine.game_over:
                moves.append(MemoryFragment(previous_state, action, state, score, True))
            else:
                moves.append(MemoryFragment(previous_state, action, state, score))
        return moves

    def _setup_tracker(
        self, data_dir: Path, description: str | None = None
    ) -> ProgressTracker:
        tracker = ProgressTracker(data_dir, to_console=True)
        tracker.write_metadata(
            dict(
                id=self.training_id,
                optimizer_factory=self._optimizer_factory.__name__,
                network_factory=self._network_factory.__name__,
                parameters=self.parameters.__dict__,
                seed=self.seed,
                description=description,
            )
        )
        return tracker

    def train(self, episodes: int, out: Path, description: str | None = None):
        self.training_id = str(uuid4())
        data_dir = out / self.training_id
        data_dir.mkdir(parents=True)
        print(f"Training with id: {self.training_id}, output directory: {data_dir}")

        tracker = self._setup_tracker(data_dir, description)

        self._initialize_networks()

        for self.episode in range(episodes):
            self.engine.reset()

            moves = self.run_episode(
                self.engine, self._select_action, self.action_to_direction
            )
            for move in moves:
                self.memory.push(move)

            tracker.track(
                episode=self.episode,
                score=self.engine.score,
                number_of_moves=len(moves),
                max_tile=max(max(moves[-1].state)),
                epsilon=self.epsilon,
            )

            self.optimize(
                device=self.device,
                loss_fn=self._loss_fn,
                memory=self.memory,
                optimizer=self.optimizer,
                parameters=self.parameters,
                policy_network=self.policy_network,
                target_network=self.target_network,
            )
            self._update_target_network()

            if self.episode > 0 and self.episode % 1000 == 0:
                self._snapshot_policy_network(data_dir)
