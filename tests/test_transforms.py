import math

import torch

from src.utils.transforms import (
    kinematics2angle,
    kinematics2vec,
)


def test_kinematics2angle_zero_heading():
    state = torch.tensor(
        [[[10.0, 20.0, 1.0, 0.0, 5.0, 0.1]]]
    )

    result = kinematics2angle(state)

    assert result.shape == (1, 1, 5)

    expected = torch.tensor(
        [[[10.0, 20.0, 0.0, 5.0, 0.1]]]
    )

    assert torch.allclose(result, expected)


def test_kinematics2angle_ninety_degrees():
    state = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0, 5.0, 0.0]]]
    )

    result = kinematics2angle(state)

    assert torch.isclose(
        result[0, 0, 2],
        torch.tensor(math.pi / 2),
    )


def test_kinematics2vec_zero_heading():
    state = torch.tensor(
        [[[10.0, 20.0, 0.0, 5.0, 0.1]]]
    )

    result = kinematics2vec(state)

    expected = torch.tensor(
        [[[10.0, 20.0, 1.0, 0.0, 5.0, 0.1]]]
    )

    assert result.shape == (1, 1, 6)
    assert torch.allclose(result, expected)


def test_angle_vector_round_trip():
    original = torch.tensor(
        [
            [
                [1.0, 2.0, 0.0, 3.0, 0.1],
                [2.0, 3.0, math.pi / 2, 4.0, 0.2],
                [3.0, 4.0, math.pi, 5.0, 0.3],
            ]
        ]
    )

    heading_vector = kinematics2vec(original)
    reconstructed = kinematics2angle(heading_vector)

     # Position, speed, and heading rate should match directly.
    assert torch.allclose(
        reconstructed[..., :2],
        original[..., :2],
        atol=1e-6,
    )

    assert torch.allclose(
        reconstructed[..., 3:],
        original[..., 3:],
        atol=1e-6,
    )

    # Angles must be compared modulo 2π.
    angle_difference = reconstructed[..., 2] - original[..., 2]

    wrapped_difference = torch.atan2(
        torch.sin(angle_difference),
        torch.cos(angle_difference),
    )

    assert torch.allclose(
        wrapped_difference,
        torch.zeros_like(wrapped_difference),
        atol=1e-6,
    )