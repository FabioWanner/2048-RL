import csv
from typing import List

import matplotlib.pyplot as plt
import numpy


def plot_running_average(name: str, values: List[int], base_line: float | None = None):
    window_width = 500
    cumulative_sum = numpy.cumsum(numpy.insert(values, 0, 0))
    running_average = (
        cumulative_sum[window_width:] - cumulative_sum[:-window_width]
    ) / window_width
    plt.figure(figsize=(12, 5))
    plt.plot(running_average)
    plt.axhline(y=base_line or float(numpy.mean(scores)), color="green", linestyle="-")
    plt.title(f"Running average of {window_width} values")
    plt.xlabel("Episode")
    plt.ylabel(name)
    plt.show()


if __name__ == "__main__":
    scores = []
    with open(
        "out/training_with_tutorial_network_but_adapted_training_procedure/data.csv",
        newline="",
    ) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            scores.append(int(row["score"]))

    plot_running_average("Score", scores, 1067)
