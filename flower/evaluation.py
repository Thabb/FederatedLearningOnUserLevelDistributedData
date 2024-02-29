from typing import List, Tuple

from flwr.common import Metrics

"""
This file holds different evaluation methods.
"""


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    In a weighted average, each data point value is multiplied by the assigned weight,
    which is then summed and divided by the number of data points.

    :param metrics: A list of numbered metrics values. Metrics values are scalar.
    :return: A new metric, that contains the accuracy of this round of evaluation.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
