from pathlib import Path

import torch

from src.model.deep_q_trainer import DeepQTrainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = DeepQTrainer(device=device)
    trainer.train(10000, Path().resolve() / "out", "Without optimizing the network")
