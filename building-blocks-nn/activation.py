import math
import torch
import numpy as np
from torch import Tensor


def naive_softmax(input_tensor: np.ndarray):
    """
    Naive approach to apply softmax to input tensor by default
    along dimension=0

    :param input_tensor:
    :return: softmax_prob
    """
    flat_tensor = input_tensor.flatten()

    softmax_prob = []
    # Find denominator first
    sum_exp = 0
    for j in range(flat_tensor.shape[0]):
        z_j = flat_tensor[j]
        sum_exp += math.exp(z_j)

    for i in range(flat_tensor.shape[0]):
        z_i = flat_tensor[i]
        softmax_i = math.exp(z_i) / sum_exp
        softmax_prob.append(softmax_i)

    return softmax_prob


def softmax(input_tensor: Tensor, dim: int) -> Tensor:
    """
    First we find the maximum value of the input tensor, then subtract it
    from each value, to avoid numer instability as exp() of a big value will be inf
    and exp() of a large negative value is 0.

    :param input_tensor:
    :param dim:
    :return:softmax_prob
    """
    # Find maximum value along the dim
    max_value = torch.max(input_tensor, dim=dim, keepdim=True).values
    # Get stable input tensor
    input_tensor_stable = input_tensor - max_value

    # Find numerator i.e. exponential of input
    numerator = torch.exp(input_tensor_stable)
    # Find denominator i.e. sum of all the input exponential
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator


def sigmoid(input_tensor: Tensor) -> Tensor:
    """
    Applies the sigmoid function to the input value
    :param input_tensor:
    :return:
    """
    return 1 / (1 + torch.exp(-input_tensor))

print(sigmoid(torch.tensor(1000.0)))
