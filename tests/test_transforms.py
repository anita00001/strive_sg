import math

import torch

from src.utils.transforms import (
    kinematics2angle,
    kinematics2vec,
)

from src.utils.transforms import (
    transform2frame
)

from src.utils.transforms import (
    pairwise_transforms
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

def test_transform_translation():
    frame = torch.tensor(
        [
            [10.0, 20.0, 0.0]
        ]
    )

    poses = torch.tensor(
        [
            [
                [13.0, 24.0, 0.0]
            ]
        ]
    )

    result = transform2frame(
        frame,
        poses,
    )

    expected = torch.tensor(
        [
            [
                [3.0, 4.0, 0.0]
            ]
        ]
    )

    assert torch.allclose(
        result,
        expected,
        atol=1e-6,
    )

def test_transform_ninety_degree_frame():
    frame = torch.tensor(
        [
            [0.0, 0.0, math.pi / 2]
        ]
    )

    poses = torch.tensor(
        [
            [
                [0.0, 5.0, math.pi / 2]
            ]
        ]
    )

    result = transform2frame(
        frame,
        poses,
    )

    expected = torch.tensor(
        [
            [
                [5.0, 0.0, 0.0]
            ]
        ]
    )

    assert torch.allclose(
        result,
        expected,
        atol=1e-6,
    )

def test_transform_relative_heading():
    frame = torch.tensor(
        [
            [0.0, 0.0, math.pi / 2]
        ]
    )

    poses = torch.tensor(
        [
            [
                [0.0, 0.0, math.pi]
            ]
        ]
    )

    result = transform2frame(
        frame,
        poses,
    )

    expected_heading = math.pi / 2

    assert torch.isclose(
        result[0, 0, 2],
        torch.tensor(expected_heading),
        atol=1e-6,
    )

def test_transform_inverse_round_trip():
    frame = torch.tensor(
        [
            [10.0, -4.0, math.pi / 3]
        ]
    )

    original = torch.tensor(
        [
            [
                [12.0, 5.0, 1.2],
                [8.0, -2.0, -0.4],
            ]
        ]
    )

    local = transform2frame(
        frame,
        original,
    )

    reconstructed = transform2frame(
        frame,
        local,
        inverse=True,
    )

    assert torch.allclose(
        reconstructed[..., :2],
        original[..., :2],
        atol=1e-5,
    )

    angle_difference = (
        reconstructed[..., 2]
        - original[..., 2]
    )

    wrapped_difference = torch.atan2(
        torch.sin(angle_difference),
        torch.cos(angle_difference),
    )

    assert torch.allclose(
        wrapped_difference,
        torch.zeros_like(wrapped_difference),
        atol=1e-5,
    )

def test_transform_heading_vector():
    frame = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0]
        ]
    )

    poses = torch.tensor(
        [
            [
                [0.0, 5.0, 0.0, 1.0]
            ]
        ]
    )

    result = transform2frame(
        frame,
        poses,
    )

    expected = torch.tensor(
        [
            [
                [5.0, 0.0, 1.0, 0.0]
            ]
        ]
    )

    assert torch.allclose(
        result,
        expected,
        atol=1e-6,
    )

# test for pairwise_transforms
def test_pairwise_transforms_shape():
    poses = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 5.0, math.pi / 2],
            ]
        ]
    )

    result = pairwise_transforms(poses)

    assert result.shape == (1, 3, 3, 3)

# test the diagonal elements of the pairwise transforms
def test_pairwise_self_transform():
    poses = torch.tensor(
        [
            [
                [2.0, 3.0, 0.5],
                [8.0, -1.0, -0.2],
            ]
        ]
    )

    result = pairwise_transforms(poses)

    diagonal = torch.stack(
        [
            result[0, 0, 0],
            result[0, 1, 1],
        ]
    )

    expected = torch.zeros_like(diagonal)

    assert torch.allclose(
        diagonal,
        expected,
        atol=1e-6,
    )

#test a simple relative position transform for pairwise_transforms
def test_pairwise_relative_position():
    poses = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ]
    )

    result = pairwise_transforms(poses)

    # Agent 1 from agent 0's frame.
    assert torch.allclose(
        result[0, 0, 1],
        torch.tensor([5.0, 0.0, 0.0]),
        atol=1e-6,
    )

    # Agent 0 from agent 1's frame.
    assert torch.allclose(
        result[0, 1, 0],
        torch.tensor([-5.0, 0.0, 0.0]),
        atol=1e-6,
    )

#test rotated frame for pairwise_transforms
def test_pairwise_rotated_frame():
    poses = torch.tensor(
        [
            [
                [0.0, 0.0, math.pi / 2],
                [0.0, 5.0, math.pi / 2],
            ]
        ]
    )

    result = pairwise_transforms(poses)

    expected = torch.tensor(
        [5.0, 0.0, 0.0]
    )

    assert torch.allclose(
        result[0, 0, 1],
        expected,
        atol=1e-6,
    )

