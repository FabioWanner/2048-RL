from pathlib import Path

import torch

from src.model.deep_q_trainer import DeepQTrainer
from src.model.networks.tutorial_network import DQNTutorial

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, "\n\n")

    trainer = DeepQTrainer(device=device, network_factory=DQNTutorial)
    trainer.train(
        10000,
        Path().resolve() / "out",
        "Training with tutorial parameters and network",
    )
