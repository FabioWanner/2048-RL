import csv
import json
from pathlib import Path
from typing import Any


class ProgressTracker:
    field_names = ["episode", "number_of_moves", "score", "max_tile", "epsilon"]

    def __init__(self, directory: Path, to_console: bool = False):
        directory.mkdir(parents=True, exist_ok=True)

        self._to_console = to_console
        self.data_file_path = directory / "data.csv"
        self.data_file_path.touch(exist_ok=False)
        self.meta_data_file_path = directory / f"meta_data.json"
        self.meta_data_file_path.touch(exist_ok=False)

        with open(self.data_file_path, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
            writer.writeheader()

    def write_metadata(self, metadata: dict[str, Any]):
        with open(self.meta_data_file_path, "w") as f:
            json.dump(metadata, f)

    def track(
        self,
        episode: int,
        score: int,
        number_of_moves: int,
        max_tile: int,
        epsilon: float,
    ):
        with open(self.data_file_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
            writer.writerow(
                dict(
                    episode=episode,
                    score=score,
                    number_of_moves=number_of_moves,
                    max_tile=max_tile,
                    epsilon=epsilon,
                )
            )

        if self._to_console:
            print(
                f"Episode {episode} finished with a score of {score}, {number_of_moves} moves were made and epsilon was: {epsilon}"
            )
