import math

import numpy as np
import torch

from src.datasets.nuscenes_utils import (
    angle_diff,
    build_category_mappings,
    heading_change_rate,
    velocity,
)

from src.datasets.nuscenes_utils import (
    align_poses_to_timeline,
    annotation_to_pose,
    build_kinematic_state,
    quaternion_yaw,
)

from src.datasets.nuscenes_utils import gen_car_coords

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

# Quaternion yaw
def test_quaternion_yaw():
    # 90 degrees around z-axis.
    rotation = [
        math.cos(math.pi / 4),
        0.0,
        0.0,
        math.sin(math.pi / 4),
    ]

    yaw = quaternion_yaw(rotation)

    assert np.isclose(
        yaw,
        math.pi / 2,
        atol=1e-6,
    )

# Annotation : heading vector
def test_annotation_to_pose():
    annotation = {
        "translation": [
            10.0,
            20.0,
            0.5,
        ],
        "rotation": [
            math.cos(math.pi / 4),
            0.0,
            0.0,
            math.sin(math.pi / 4),
        ],
    }

    pose = annotation_to_pose(
        annotation
    )

    expected = np.array(
        [
            10.0,
            20.0,
            0.0,
            1.0,
        ]
    )

    assert np.allclose(
        pose,
        expected,
        atol=1e-6,
    )

#  Timeline allignment
def test_align_poses_to_timeline():
    timeline = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
        ]
    )

    observation_times = np.array(
        [
            0.5,
            1.0,
        ]
    )

    poses = np.array(
        [
            [2.0, 0.0, 1.0, 0.0],
            [4.0, 0.0, 1.0, 0.0],
        ]
    )

    aligned = align_poses_to_timeline(
        observation_times,
        poses,
        timeline,
    )

    assert aligned.shape == (
        4,
        4,
    )

    assert np.isnan(
        aligned[0]
    ).all()

    assert np.allclose(
        aligned[1],
        poses[0],
    )

    assert np.allclose(
        aligned[2],
        poses[1],
    )

    assert np.isnan(
        aligned[3]
    ).all()

# Full Six - Dimensional Kinematic state
def test_build_kinematic_state():
    timestamps = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
        ]
    )

    poses = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
            [4.0, 0.0, 1.0, 0.0],
            [6.0, 0.0, 1.0, 0.0],
        ]
    )

    state, visibility = (
        build_kinematic_state(
            poses,
            timestamps,
        )
    )

    assert state.shape == (
        4,
        6,
    )

    # x, y
    assert np.allclose(
        state[:, :2],
        poses[:, :2],
    )

    # hx, hy
    assert np.allclose(
        state[:, 2:4],
        poses[:, 2:4],
    )

    # 2 meters every 0.5 sec = 4 m/s
    assert np.allclose(
        state[:, 4],
        4.0,
    )

    # Constant heading => zero heading rate.
    assert np.allclose(
        state[:, 5],
        0.0,
        atol=1e-6,
    )

    assert np.all(
        visibility == 1.0
    )

# Missing trajectory frames
def test_build_kinematic_state_with_missing_frames():
    timestamps = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
        ]
    )

    poses = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
            [4.0, 0.0, 1.0, 0.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )

    state, visibility = (
        build_kinematic_state(
            poses,
            timestamps,
        )
    )

    assert np.isnan(
        state[0]
    ).all()

    assert np.isnan(
        state[-1]
    ).all()

    assert visibility[0] == 0.0
    assert visibility[-1] == 0.0

    assert visibility[1] == 1.0
    assert visibility[2] == 1.0

# test heading east
def test_gen_car_coords_heading_east():
    positions = torch.tensor(
        [
            [10.0, 20.0]
        ]
    )

    headings = torch.tensor(
        [
            [1.0, 0.0]
        ]
    )

    coords = gen_car_coords(
        positions,
        headings,
        num_channels=1,
        length_pixels=3,
        width_pixels=3,
        bounds=[
            -1.0,
            -1.0,
            1.0,
            1.0,
        ],
    )

    assert coords.shape == (
        1,
        1,
        3,
        3,
        2,
    )

    # Center pixel must coincide with the agent.
    assert torch.allclose(
        coords[0, 0, 1, 1],
        torch.tensor(
            [10.0, 20.0]
        ),
        atol=1e-6,
    )

    # One meter forward.
    assert torch.allclose(
        coords[0, 0, 2, 1],
        torch.tensor(
            [11.0, 20.0]
        ),
        atol=1e-6,
    )

# test heading north
def test_gen_car_coords_heading_north():
    positions = torch.tensor(
        [
            [0.0, 0.0]
        ]
    )

    headings = torch.tensor(
        [
            [0.0, 1.0]
        ]
    )

    coords = gen_car_coords(
        positions,
        headings,
        num_channels=1,
        length_pixels=3,
        width_pixels=3,
        bounds=[
            -1.0,
            -1.0,
            1.0,
            1.0,
        ],
    )

    # One meter forward in local coordinates
    # becomes +1 meter in global y.
    assert torch.allclose(
        coords[0, 0, 2, 1],
        torch.tensor(
            [0.0, 1.0]
        ),
        atol=1e-6,
    )

    # One meter in +local lateral direction
    # becomes -1 meter in global x.
    assert torch.allclose(
        coords[0, 0, 1, 2],
        torch.tensor(
            [-1.0, 0.0]
        ),
        atol=1e-6,
    )

# test multiple map channels
def test_gen_car_coords_channels_share_geometry():
    positions = torch.tensor(
        [
            [5.0, 7.0]
        ]
    )

    headings = torch.tensor(
        [
            [1.0, 0.0]
        ]
    )

    coords = gen_car_coords(
        positions,
        headings,
        num_channels=4,
        length_pixels=3,
        width_pixels=3,
        bounds=[
            -1.0,
            -1.0,
            1.0,
            1.0,
        ],
    )

    assert coords.shape == (
        1,
        4,
        3,
        3,
        2,
    )

    # Every semantic raster channel uses the same
    # world-space sampling grid.
    assert torch.allclose(
        coords[:, 0],
        coords[:, 1],
    )

    assert torch.allclose(
        coords[:, 0],
        coords[:, 2],
    )

    assert torch.allclose(
        coords[:, 0],
        coords[:, 3],
    )

# test vehicle-footprint mode
def test_gen_car_coords_vehicle_dimensions():
    positions = torch.tensor(
        [
            [0.0, 0.0]
        ]
    )

    headings = torch.tensor(
        [
            [1.0, 0.0]
        ]
    )

    coords = gen_car_coords(
        positions,
        headings,
        num_channels=1,
        length_pixels=3,
        width_pixels=3,
        lengths=torch.tensor(
            [4.0]
        ),
        widths=torch.tensor(
            [2.0]
        ),
    )

    # Front center:
    assert torch.allclose(
        coords[0, 0, 2, 1],
        torch.tensor(
            [2.0, 0.0]
        ),
        atol=1e-6,
    )

    # Left/right extent:
    assert torch.allclose(
        coords[0, 0, 1, 2],
        torch.tensor(
            [0.0, 1.0]
        ),
        atol=1e-6,
    )

