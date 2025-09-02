from pathlib import Path

import torch

from src.model.deep_q_trainer import DeepQTrainer
from src.model.networks.convolutional_network import DQNConvolutional
from src.model.parameters import Parameters


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, "\n\n")

    parameters = Parameters(
        batch_size=512,
        learning_rate=5e-5,
        replay_memory_size=1024 * 20,
        epsilon_decay_rate=1000,
    )

    trainer = DeepQTrainer(
        device=device, parameters=parameters, network_factory=DQNConvolutional
    )

    trainer.train(
        30000,
        Path().resolve() / "out",
        "Training with small convolutional network",
    )
