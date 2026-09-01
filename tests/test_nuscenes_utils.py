import math

import numpy as np
import torch

from src.datasets.nuscenes_utils import (
    angle_diff,
    build_category_mappings,
    heading_change_rate,
    velocity,
)

# category mapping
def test_category_mapping():
    (
        categories,
        key2cat,
        cat2vec,
        vec2cat,
    ) = build_category_mappings(
        ["car", "truck"]
    )

    assert categories == [
        "car",
        "truck",
    ]

    assert key2cat["vehicle.car"] == "car"
    assert key2cat["vehicle.truck"] == "truck"

    assert torch.equal(
        cat2vec["car"],
        torch.tensor([1, 0]),
    )

    assert torch.equal(
        cat2vec["truck"],
        torch.tensor([0, 1]),
    )

    assert vec2cat[(1, 0)] == "car"
    assert vec2cat[(0, 1)] == "truck"

# Reduced categories
def test_reduced_category_mapping():
    (
        categories,
        key2cat,
        _,
        _,
    ) = build_category_mappings(
        [
            "car",
            "truck",
            "bus",
            "trailer",
            "emergency",
            "construction",
        ],
        reduce_cats=True,
    )

    assert categories == [
        "car",
        "truck",
    ]

    assert key2cat["vehicle.car"] == "car"
    assert key2cat["vehicle.emergency"] == "car"

    assert key2cat["vehicle.truck"] == "truck"
    assert key2cat["vehicle.bus"] == "truck"
    assert key2cat["vehicle.trailer"] == "truck"
    assert key2cat["vehicle.construction"] == "truck"

# Important bicycle behaviour
def test_reduced_categories_keep_motorcycle_separate():
    categories, _, _, _ = build_category_mappings(
        [
            "car",
            "motorcycle",
            "cyclist",
        ],
        reduce_cats=True,
    )

    assert categories == [
        "car",
        "cyclist",
        "motorcycle",
    ]

# Angular wrap
def test_angle_diff_wraps_across_pi():
    theta1 = np.array(
        [
            math.radians(-179.0)
        ]
    )

    theta2 = np.array(
        [
            math.radians(179.0)
        ]
    )

    result = angle_diff(
        theta1,
        theta2,
    )

    assert np.allclose(
        result,
        np.array(
            [
                math.radians(2.0)
            ]
        ),
        atol=1e-6,
    )

# constant velocity
def test_velocity_constant_motion():
    timestamps = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
        ]
    )

    positions = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ]
    )

    result = velocity(
        positions,
        timestamps,
    )

    expected = np.array(
        [
            [4.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
        ]
    )

    assert np.allclose(
        result,
        expected,
    )

# constant heading angle
def test_heading_change_rate_constant():
    timestamps = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
        ]
    )

    headings = np.array(
        [
            0.0,
            0.1,
            0.2,
            0.3,
        ]
    )

    result = heading_change_rate(
        headings,
        timestamps,
    )

    expected = np.array(
        [
            0.2,
            0.2,
            0.2,
            0.2,
        ]
    )

    assert np.allclose(
        result,
        expected,
        atol=1e-6,
    )

