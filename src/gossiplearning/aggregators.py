import keras
import numpy as np
import torch
from torch import nn
from laplace import Laplace

from gossiplearning.models import (
    MergeStrategy,
    AggregatorFn,
    WeightsMessage,
)
from gossiplearning.weights_marshaling import unflatten_weights, flatten_weights


def merge_weights_with_simple_avg(
    current_model: nn.Module,
    node_model_age: int,
    weights_messages: tuple[WeightsMessage, ...],
) -> tuple[nn.Module, int]:
    """
    Aggregate weights performing a simple arithmetic average.

    For each single weight, perform the arithmetic average between the current weight and the
    received weights. If no weight was received for some index, keep the current model weight.

    :param current_model: the current model
    :param node_model_age: the age of the current model
    :param weights_messages: the marshaled received model weights
    :return: the new weights
    """
    model_weights = flatten_weights(current_model)
    n_weights = len(model_weights)
    counters = np.zeros(n_weights)
    sums = np.zeros(n_weights)

    for msg in weights_messages:
        mw = msg.marshaled_weights
        counters[mw.indices] += 1
        sums[mw.indices] += mw.weights

    counters += 1
    sums += model_weights

    if any(counters == 0):
        raise Exception("No weight provided for some indexes!")

    final_weights = unflatten_weights(current_model, sums / counters)

    
    with torch.no_grad():
        for param, new_val in zip(current_model.parameters(), final_weights):
            param.copy_(new_val)

    new_model_age = max(
        node_model_age,
        *[mw.model_weight for mw in weights_messages],
    )

    return current_model, new_model_age


def merge_weights_with_age_weighted_avg(
    current_model: nn.Module,
    node_model_age: int,
    weights_messages: tuple[WeightsMessage, ...],
) -> tuple[nn.Module, int]:
    """
    Aggregate current model weights and received weights averaging them based on model age.

    Weighted-average giving more importance to more recent models. Marshaled weights include
    the model age, which is the last time when the model was trained. The higher such time,
    the more importance the related weight gets.

    If no weight was received for some index, keep the current model weight.

    :param current_model: the current model
    :param node_model_age: the age of the current model
    :param weights_messages: the marshaled received model weights
    :return: the new weights
    """
    model_weights = flatten_weights(current_model)
    n_weights = len(model_weights)
    ages_sum = np.zeros(n_weights)
    sums = np.zeros(n_weights)

    for msg in weights_messages:
        mw = msg.marshaled_weights
        ages_sum[mw.indices] += msg.model_weight if msg.model_weight > 0 else 1
        sums[mw.indices] += mw.weights * (
            msg.model_weight if msg.model_weight > 0 else 1
        )

    ages_sum += node_model_age
    sums += model_weights * (node_model_age if node_model_age > 0 else 1)

    # if current model age is 0 and no weight was received for this index
    ages_sum[ages_sum == 0] = 1

    final_weights = unflatten_weights(current_model, sums / ages_sum)
    
    with torch.no_grad():
        for param, new_val in zip(current_model.parameters(), final_weights):
            param.copy_(new_val)

    new_model_age = max(
        node_model_age,
        *[mw.model_weight for mw in weights_messages],
    )

    return current_model, new_model_age


def merge_weights_with_overwrite(
    current_model: nn.Module,
    node_model_age: int,
    weights_messages: tuple[WeightsMessage, ...],
) -> tuple[nn.Module, int]:
    """
    Aggregate model weights by overwriting the current model ones.

    This merge strategy can only be applied if the number of models to be merged is 2 (the current one and one received
    model) and if no compression is applied to messages.

    :param current_model: the current model
    :param node_model_age: the age of the current model
    :param weights_messages: the marshaled received model weights
    :return: the new weights
    """
    if len(weights_messages) != 1:
        raise RuntimeError(
            "MergeNone strategy only works with one single received model!"
        )

    received_weights = unflatten_weights(
        current_model, weights_messages[0].marshaled_weights.weights
    )

    #current_model.set_weights(received_weights)

    with torch.no_grad():
        for param, new_val in zip(current_model.parameters(), received_weights):
            param.copy_(new_val)

    if opt_state := weights_messages[0].optimizer_state:
        current_model.optimizer.build(current_model.variables)
        current_model.optimizer.set_weights(opt_state)

    return current_model, weights_messages[0].model_weight


def merge_with_intelligent_overwrite(
    current_model: nn.Module,
    node_model_age: int,
    weights_messages: tuple[WeightsMessage, ...],
) -> tuple[nn.Module, int]:
    """
    Aggregate model weights by overwriting the current model ones.

    The probability of overwriting the current model depends on the two model ages, giving to the older models more
    chances of being kept.

    This merge strategy can only be applied if the number of models to be merged is 2 (the current one and one received
    model) and if no compression is applied to messages.

    :param current_model: the current model
    :param node_model_age: the age of the current model
    :param weights_messages: the marshaled received model weights
    :return: the new weights
    """
    if len(weights_messages) != 1:
        raise RuntimeError(
            "MergeNone strategy only works with one single received model!"
        )

    overwrite_model = node_model_age == 0

    received_age = weights_messages[0].model_weight
    if not overwrite_model and received_age > 0:
        if received_age > node_model_age:
            ratio = received_age / node_model_age
            prob = 1 / (ratio**2)
            overwrite_model = np.random.choice([True, False], p=[1 - prob, prob])
        else:
            ratio = node_model_age / received_age
            prob = 1 / (ratio**2)
            overwrite_model = np.random.choice([True, False], p=[prob, 1 - prob])

    if overwrite_model:
        received_weights = unflatten_weights(
            current_model, weights_messages[0].marshaled_weights.weights
        )

        #current_model.set_weights(received_weights)

        with torch.no_grad():
            for param, new_val in zip(current_model.parameters(), received_weights):
                param.copy_(new_val)

        if opt_state := weights_messages[0].optimizer_state:
            current_model.optimizer.build(current_model.variables)
            current_model.optimizer.set_weights(opt_state)

    return (
        current_model,
        weights_messages[0].model_weight if overwrite_model else node_model_age,
    )

def OLD_merge_with_fisher_laplace_method(
    current_model: nn.Module,
    node_model_age: int,
    weights_messages: tuple[WeightsMessage, ...],
    fisher_messages,
    received_fisher,
    lambdas=None
)-> tuple[nn.Module, int]:

    model_weights = flatten_weights(current_model)
    n = len(model_weights)
    m = len(weights_messages)
    num = np.zeros(n)
    den = np.zeros(n)

    if lambdas is None:
        lambdas = np.ones(m) / m

    f = np.diag(fisher_messages)
    for λ, w_msg, in zip(lambdas, weights_messages):

        w = w_msg.marshaled_weights
        #f = np.diag(fisher_messages)
        idx = w.indices
        num[idx] += λ * f[idx] * w.weights
        den[idx] += λ * f[idx]

    merged = np.where(den > 0, num / den, model_weights)
    final_weights = unflatten_weights(current_model, merged)

    with torch.no_grad():
        for p, nw in zip(current_model.parameters(), final_weights):
            p.copy_(nw)

    new_model_age = max(
        node_model_age,
        *[mw.model_weight for mw in weights_messages],
    )
    return current_model, new_model_age

def merge_with_fisher_laplace_method(
    current_model: nn.Module,
    node_model_age: int,
    weights_messages: tuple[WeightsMessage, ...],
    fisher_messages,
    received_fisher
)-> tuple[nn.Module, int]:

    w_local = flatten_weights(current_model)
    n = len(w_local)
    m = len(weights_messages)
    num = np.zeros(n)
    den = np.zeros(n)
    received_age = weights_messages[0].model_weight

    """if received_age > node_model_age:
        ratio = received_age / node_model_age
        prob = 1 / (ratio**2)
    else:
        ratio = node_model_age / received_age
        prob = 1 / (ratio**2)"""

    if type(fisher_messages) is not np.ndarray:
        F_local = fisher_messages.numpy()
    else:
        F_local = fisher_messages
    if type(received_fisher) is not np.ndarray:
        F_received = received_fisher.numpy()
    else:
        F_received = received_fisher

    for w_msg, in zip(weights_messages):

        w = w_msg.marshaled_weights
        idx = w.indices

        """if received_age > node_model_age:
            num[idx] += prob * F_local[idx] * w_local[idx] + (1 - prob) * F_received[idx] * w.weights[idx]
            den[idx] += prob * F_local[idx] + (1 - prob) * F_received[idx]
        else:
            num[idx] += (1 - prob) * F_local[idx] * w_local[idx] + prob * F_received[idx] * w.weights[idx]
            den[idx] += (1 - prob) * F_local[idx] + prob * F_received[idx]"""
        
        num[idx] += F_local[idx] * w_local[idx] + F_received[idx] * w.weights[idx]
        den[idx] += F_local[idx] + F_received[idx]

    merged = np.where(den > 0, num / den, w_local)
    final = unflatten_weights(current_model, merged)

    with torch.no_grad():
        for p, nw in zip(current_model.parameters(), final):
            p.copy_(nw)

    intermediate_F = F_local + F_received

    new_model_age = received_age + node_model_age
    return current_model, new_model_age, intermediate_F


def merge_with_custom_strategy(fn: AggregatorFn) -> AggregatorFn:
    def wrapped_fn(
        current_model: nn.Module,
        model_age: int,
        models_weights: tuple[WeightsMessage, ...],
    ) -> tuple[nn.Module, int]:
        res = fn(current_model, model_age, models_weights)
        return res

    return wrapped_fn


def choose_aggregator(
    strategy: MergeStrategy = MergeStrategy.SIMPLE_AVG,
    custom_fn: AggregatorFn | None = None,
) -> AggregatorFn:
    if custom_fn:
        return merge_with_custom_strategy(custom_fn)

    handlers = {
        MergeStrategy.SIMPLE_AVG: merge_weights_with_simple_avg,
        MergeStrategy.AGE_WEIGHTED: merge_weights_with_age_weighted_avg,
        MergeStrategy.OVERWRITE: merge_weights_with_overwrite,
        MergeStrategy.IMPROVED_OVERWRITE: merge_with_intelligent_overwrite,
        MergeStrategy.FISHER_LAPLACE: merge_with_fisher_laplace_method,
    }

    if strategy not in handlers:
        raise Exception("Unrecognized merge strategy!")

    return handlers[strategy]
