import os

from experiment.runner import (
    run_experiment
)


if __name__ == "__main__":

    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "experiment",
        "dataset.json"
    )

    run_experiment(
        dataset_path
    )