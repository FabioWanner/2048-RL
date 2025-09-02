from pathlib import Path
from typing import Callable

import torch

from src.engine.engine import Engine2048, Direction
from src.model.networks.convolutional_network import DQNConvolutional
from src.model.networks.base import Network


class Play2048:
    network_action_to_direction = {
        0: Direction.UP,
        1: Direction.DOWN,
        2: Direction.LEFT,
        3: Direction.RIGHT,
    }

    network_action_to_user_action = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
    }

    user_action_to_direction = {
        "w": Direction.UP,
        "s": Direction.DOWN,
        "a": Direction.LEFT,
        "d": Direction.RIGHT,
    }

    def __init__(
        self,
        board_size: int = 4,
        engine_factory: Callable[[int, int | None], Engine2048] = Engine2048,
        network_factory: Callable[
            [int, int, torch.device | None], Network
        ] = DQNConvolutional,
        network_parameters: Path | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
    ):
        self.engine = engine_factory(board_size, seed)
        self.network = network_factory(board_size, 4, device).to(device)

        if network_parameters:
            self.network.load_state_dict(torch.load(network_parameters))

    def _get_action(self) -> int:
        return self.network([self.engine.state]).argmax().item()

    def _display_board(self):
        print("\n")
        print(f"Score: {self.engine.score}")
        [print(row) for row in self.engine.state]

    def start(self):
        print("Welcome to 2048!\n")
        print(
            "Type 'a' for left, 's' for down, 'd' for right, 'w' for up or 'h' for help and press ENTER to confirm. \n\n"
        )

        self._display_board()

        while not self.engine.game_over:
            action = input("")
            if action in self.user_action_to_direction:
                self.engine.evolve(self.user_action_to_direction[action])
                self.engine.spawn()
                self._display_board()
            elif action == "h":
                suggestion = self._get_action()
                print(
                    f"The AI suggest to go: ",
                    self.network_action_to_user_action[suggestion],
                )
                continue
        print("Game Over!\n")
        input("Press enter to restart.")
        self.start()


if __name__ == "__main__":
    game = Play2048(
        network_parameters=Path().resolve()
        / "out/training_with_small_convolutional_network/model.pt"
    )
    game.start()
