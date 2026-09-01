import numpy as np
import torch
from torch import nn

from src.utils.torch import (
    c2c,
    calc_conv_out,
    compute_kl_weight,
    count_params,
    get_device,
    tensor_clamp,
)


def test_c2c():
    tensor = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    array = c2c(tensor)

    assert isinstance(array, np.ndarray)
    assert np.allclose(
        array,
        np.array([1.0, 2.0, 3.0]),
    )


def test_get_device():
    device = get_device()

    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_count_params():
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.Linear(3, 2),
    )

    result = count_params(model)

    # First layer:
    #   weights = 4 * 3 = 12
    #   bias    = 3
    #
    # Second layer:
    #   weights = 3 * 2 = 6
    #   bias    = 2
    #
    # Total = 23
    assert result == 23


def test_count_params_ignores_frozen_parameters():
    model = nn.Linear(4, 3)

    model.bias.requires_grad = False

    result = count_params(model)

    assert result == 12


def test_calc_conv_out():
    result = calc_conv_out(
        in_size=256,
        kernel_size=7,
        stride=2,
    )

    assert result == 125


def test_calc_conv_out_with_padding():
    result = calc_conv_out(
        in_size=32,
        kernel_size=3,
        stride=1,
        padding_size=1,
    )

    assert result == 32


def test_compute_kl_weight_start():
    result = compute_kl_weight(
        cur_epoch=0,
        end_epoch=20,
        final_kl_weight=0.004,
    )

    assert result == 0.0


def test_compute_kl_weight_halfway():
    result = compute_kl_weight(
        cur_epoch=10,
        end_epoch=20,
        final_kl_weight=0.004,
    )

    assert result == 0.002


def test_compute_kl_weight_caps_at_final_weight():
    result = compute_kl_weight(
        cur_epoch=50,
        end_epoch=20,
        final_kl_weight=0.004,
    )

    assert result == 0.004


def test_tensor_clamp_scalar_bounds():
    values = torch.tensor(
        [-2.0, 0.5, 5.0]
    )

    result = tensor_clamp(
        values,
        xmin=0.0,
        xmax=2.0,
    )

    expected = torch.tensor(
        [0.0, 0.5, 2.0]
    )

    assert torch.allclose(
        result,
        expected,
    )


def test_tensor_clamp_tensor_bounds():
    values = torch.tensor(
        [-1.0, 4.0, 8.0]
    )

    lower = torch.tensor(
        [0.0, 2.0, 5.0]
    )

    upper = torch.tensor(
        [1.0, 3.0, 6.0]
    )

    result = tensor_clamp(
        values,
        lower,
        upper,
    )

    expected = torch.tensor(
        [0.0, 3.0, 6.0]
    )

    assert torch.allclose(
        result,
        expected,
    )
    