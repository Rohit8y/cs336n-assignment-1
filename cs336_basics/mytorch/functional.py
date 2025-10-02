import torch
from torch import Tensor

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