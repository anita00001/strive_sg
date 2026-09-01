import numpy as np
import torch
from torch import nn


def c2c(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a detached NumPy array on CPU.

    Historically STRIVE calls this helper `c2c`, meaning roughly
    CUDA-to-CPU.
    """
    return tensor.detach().cpu().numpy()


def get_device(device_index: int = 0) -> torch.device:
    """
    Return a CUDA device when CUDA is available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")

    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def calc_conv_out(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding_size: int = 0,
    dilation: int = 1,
) -> int:
    """
    Calculate the spatial output size of one convolution dimension.

    Formula:

        floor(
            (input + 2*padding
             - dilation*(kernel-1)
             - 1)
            / stride
            + 1
        )
    """
    numerator = (
        in_size
        + 2 * padding_size
        - dilation * (kernel_size - 1)
        - 1
    )

    return numerator // stride + 1


def compute_kl_weight(
    cur_epoch: int,
    end_epoch: int,
    final_kl_weight: float,
) -> float:
    """
    Linear KL annealing.

    Weight starts at zero and reaches `final_kl_weight`
    at `end_epoch`.
    """
    if end_epoch <= 0:
        return final_kl_weight

    progress = cur_epoch / end_epoch

    progress = max(
        0.0,
        min(1.0, progress),
    )

    return progress * final_kl_weight


def tensor_clamp(
    x: torch.Tensor,
    xmin: torch.Tensor | float,
    xmax: torch.Tensor | float,
) -> torch.Tensor:
    """
    Clamp a tensor when minimum and maximum bounds may themselves
    be tensors.
    """
    xmin = torch.as_tensor(
        xmin,
        dtype=x.dtype,
        device=x.device,
    )

    xmax = torch.as_tensor(
        xmax,
        dtype=x.dtype,
        device=x.device,
    )

    return torch.maximum(
        torch.minimum(x, xmax),
        xmin,
    )
