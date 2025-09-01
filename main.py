from pathlib import Path
from time import sleep

import torch

from src.model.deep_q_trainer import DeepQTrainer
from src.model.networks.tutorial_network import DQNTutorial

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, "\n\n")

    trainer = DeepQTrainer(device=device, network_factory=DQNTutorial)
    network_state_path = None

    retries = 10
    while retries > 0:
        try:
            trainer.train(
                10000,
                Path().resolve() / "out",
                "Training with tutorial network and penalties",
                network_state_path,
            )
        except Exception as e:
            print(e)
            sleep(30)
            network_state_path = (
                Path().resolve() / "out" / trainer.training_id / "model.pt"
            )
            if not network_state_path.exists():
                network_state_path = None
            retries -= 1
