from pathlib import Path
from time import sleep

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
    network_state_path = None

    retries = 10
    while retries > 0:
        try:
            trainer.train(
                30000,
                Path().resolve() / "out",
                "Training with small convolutional network",
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
