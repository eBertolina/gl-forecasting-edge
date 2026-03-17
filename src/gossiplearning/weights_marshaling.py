import math
from typing import Callable

import keras
import numpy as np
import torch
from torch import nn

from gossiplearning.models import MarshaledWeights, FlattenedWeights, ModelWeights

MarshalWeightsFn = Callable[[keras.Model, float], MarshaledWeights]


#def flatten_weights(model: keras.Model) -> FlattenedWeights:
#    num_weights = model.count_params()
#    weights = np.empty(num_weights)#

#    start = 0
#    for l in model.get_weights():
#        flattened = l.flatten()
#        end = start + len(flattened)
#        weights[start:end] = flattened
#        start = end

#    return weights


def flatten_weights(model: nn.Module) -> FlattenedWeights:
    num_weights = sum(p.numel() for p in model.parameters())
    weights = np.empty(num_weights, dtype=np.float32)
    
    start = 0
    for param in model.parameters():
        flattened = param.detach().cpu().numpy().flatten()
        end = start + flattened.size
        weights[start:end] = flattened
        start = end
        
    return weights



def marshal_weights_with_random_subsampling(
    model: nn.Module, perc_weights: float = 1      #keras.model
) -> MarshaledWeights:
    flattened_weights = flatten_weights(model)

    n_weights = len(flattened_weights)
    indices = np.arange(0, n_weights)

    if perc_weights == 1:
        return MarshaledWeights(
            indices=indices,
            weights=flattened_weights,
        )

    n_samples = np.int64(perc_weights * n_weights)
    selected_indices = np.random.choice(indices, size=n_samples, replace=False)

    return MarshaledWeights(
        indices=selected_indices,
        weights=flattened_weights[selected_indices],
    )


def unflatten_weights_original(
    model_blueprint: keras.Model, flattened: FlattenedWeights
) -> ModelWeights:
    """
    Transform a set of flattened weights into proper layer-wise model weights.
    :param flattened: the flattened weights
    :return: the model weights group by layer
    """
    res: ModelWeights = []

    start = 0
    for l in model_blueprint.get_weights():
        end = start + math.prod(l.shape)
        weights = flattened[start:end]
        weights = weights.reshape(l.shape)

        res = res + [weights]

        start = end

    return res


def unflatten_weights(
        model_blueprint: nn.Module,
        flattened: FlattenedWeights) -> ModelWeights:
    """
    Transform a set of flattened weights into proper layer-wise model weights.
    :param flattened: the flattened weights
    :return: the model weights group by layer
    """
    res: ModelWeights = []

    start = 0
    for param in model_blueprint.parameters():
        shape = param.shape
        numel = param.numel()

        end = start + numel
        weights = flattened[start:end]
        weights = weights.reshape(shape)

        tensor = torch.tensor(weights, dtype=param.dtype)
        res.append(tensor)

        start = end

    return res

