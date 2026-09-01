# category mapping
import numpy as np
import torch

from pyquaternion import Quaternion   # yaw


NUSC_CATEGORY_KEYS = {
    "car": ["vehicle.car"],
    "truck": ["vehicle.truck"],
    "bus": ["vehicle.bus"],
    "motorcycle": ["vehicle.motorcycle"],
    "trailer": ["vehicle.trailer"],
    "cyclist": ["vehicle.bicycle"],
    "pedestrian": ["human.pedestrian"],
    "emergency": ["vehicle.emergency"],
    "construction": ["vehicle.construction"],
}


NUSC_REDUCED_CATEGORY_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus": "truck",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "truck",
    "vehicle.bicycle": "cyclist",
    "human.pedestrian": "pedestrian",
    "vehicle.emergency": "car",
    "vehicle.construction": "truck",
}


def build_category_mappings(
    categories: list[str],
    *,
    reduce_cats: bool = False,
):
    """
    Build STRIVE's nuScenes category mappings.

    Returns
    -------
    categories:
        Final model category names.

    key2cat:
        Maps raw nuScenes category names to model categories.

    cat2vec:
        Maps model categories to one-hot tensors.

    vec2cat:
        Reverse mapping from one-hot tuples to category names.
    """
    unknown = [
        category
        for category in categories
        if category not in NUSC_CATEGORY_KEYS
    ]

    if unknown:
        raise ValueError(
            f"Unrecognized categories: {unknown}"
        )

    key2cat = {}

    for category in categories:
        for key in NUSC_CATEGORY_KEYS[category]:
            key2cat[key] = category

    if reduce_cats:
        key2cat = {
            key: NUSC_REDUCED_CATEGORY_MAP[key]
            for key in key2cat
        }

        final_categories = sorted(
            set(key2cat.values())
        )

    else:
        # Preserve requested order, matching STRIVE.
        final_categories = list(categories)

    identity = torch.eye(
        len(final_categories),
        dtype=torch.int64,
    )

    cat2vec = {
        category: identity[index]
        for index, category
        in enumerate(final_categories)
    }

    vec2cat = {
        tuple(identity[index].tolist()): category
        for index, category
        in enumerate(final_categories)
    }

    return (
        final_categories,
        key2cat,
        cat2vec,
        vec2cat,
    )

# Angular differences
# +179° and -179° should differ by roughly: 2°
def angle_diff(
    theta1: np.ndarray,
    theta2: np.ndarray,
) -> np.ndarray:
    """
    Return the signed smallest angular difference theta1 - theta2.

    Result lies in [-pi, pi).
    """
    return (
        theta1
        - theta2
        + np.pi
    ) % (2.0 * np.pi) - np.pi

# Velocity
'''
    For constant-speed motion:
        t       0    0.5    1.0    1.5
        x       0     2      4      6
    we get:
        vx = 2/0.5 = 4 m/s
'''
def velocity(
    positions: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Estimate velocity from positions using finite differences.

    Parameters
    ----------
    positions:
        Shape (T, D)

    timestamps:
        Shape (T,), in seconds.

    Returns
    -------
    np.ndarray:
        Velocity of shape (T, D).

    Notes
    -----
    STRIVE primarily uses backward differences, with a forward
    difference for the first valid timestep.
    """
    positions = np.asarray(
        positions,
        dtype=np.float64,
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    if positions.ndim != 2:
        raise ValueError(
            "positions must have shape (T, D)"
        )

    if timestamps.ndim != 1:
        raise ValueError(
            "timestamps must have shape (T,)"
        )

    if positions.shape[0] != timestamps.shape[0]:
        raise ValueError(
            "positions and timestamps must have equal length"
        )

    num_steps = positions.shape[0]

    if num_steps < 2:
        return np.full_like(
            positions,
            np.nan,
            dtype=np.float64,
        )

    dt = np.diff(timestamps)

    if np.any(dt <= 0):
        raise ValueError(
            "timestamps must be strictly increasing"
        )

    differences = (
        positions[1:] - positions[:-1]
    ) / dt[:, None]

    result = np.concatenate(
        [
            differences[:1],
            differences,
        ],
        axis=0,
    )

    # If a valid point follows a NaN point, use its forward
    # difference instead when available.
    valid = ~np.isnan(positions).any(axis=1)

    for index in range(1, num_steps):
        if valid[index] and not valid[index - 1]:
            if index + 1 < num_steps and valid[index + 1]:
                result[index] = (
                    positions[index + 1]
                    - positions[index]
                ) / (
                    timestamps[index + 1]
                    - timestamps[index]
                )

    return result

# Heading change rate
def heading_change_rate(
    headings: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Estimate heading angular rate using wrapped finite differences.

    Parameters
    ----------
    headings:
        Shape (T,), radians.

    timestamps:
        Shape (T,), seconds.
    """
    headings = np.asarray(
        headings,
        dtype=np.float64,
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    if headings.ndim != 1:
        raise ValueError(
            "headings must have shape (T,)"
        )

    if timestamps.ndim != 1:
        raise ValueError(
            "timestamps must have shape (T,)"
        )

    if headings.shape[0] != timestamps.shape[0]:
        raise ValueError(
            "headings and timestamps must have equal length"
        )

    num_steps = headings.shape[0]

    if num_steps < 2:
        return np.full_like(
            headings,
            np.nan,
            dtype=np.float64,
        )

    dt = np.diff(timestamps)

    if np.any(dt <= 0):
        raise ValueError(
            "timestamps must be strictly increasing"
        )

    differences = angle_diff(
        headings[1:],
        headings[:-1],
    ) / dt

    result = np.concatenate(
        [
            differences[:1],
            differences,
        ]
    )

    valid = ~np.isnan(headings)

    for index in range(1, num_steps):
        if valid[index] and not valid[index - 1]:
            if index + 1 < num_steps and valid[index + 1]:
                result[index] = (
                    angle_diff(
                        np.asarray([headings[index + 1]]),
                        np.asarray([headings[index]]),
                    )[0]
                    / (
                        timestamps[index + 1]
                        - timestamps[index]
                    )
                )

    return result

# Yaw from quaternion
def quaternion_yaw(
    rotation,
) -> float:
    """
    Extract planar yaw from a nuScenes quaternion.

    nuScenes stores orientation as:
        [w, x, y, z]
    """
    rotation_matrix = Quaternion(
        rotation
    ).rotation_matrix

    return float(
        np.arctan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0],
        )
    )


# convert one annotation to planar pose
'''
    car pointing 90 degrees has:
        h= pi/2
        therefore, (hx, hy) = (cos(h), sin(h)) = (0,1)
'''
def annotation_to_pose(
    annotation: dict,
) -> np.ndarray:
    """
    Convert a nuScenes annotation to planar STRIVE pose.

    Returns
    -------
    np.ndarray
        [x, y, hx, hy]
    """
    translation = np.asarray(
        annotation["translation"],
        dtype=np.float64,
    )

    if translation.shape[0] < 2:
        raise ValueError(
            "annotation translation must contain x and y"
        )

    yaw = quaternion_yaw(
        annotation["rotation"]
    )

    return np.array(
        [
            translation[0],
            translation[1],
            np.cos(yaw),
            np.sin(yaw),
        ],
        dtype=np.float64,
    )

# Align an incomplete agent trajectory
# The ego vehicle exists at every scene timestamp,
# but another agent may only exist at some frames.
# Use NaN to fill in missing positions and headings.
def align_poses_to_timeline(
    observation_times: np.ndarray,
    poses: np.ndarray,
    timeline: np.ndarray,
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Align observed poses to a common scene timeline.

    Missing frames are represented by NaN.

    Parameters
    ----------
    observation_times:
        Shape (K,), seconds.

    poses:
        Shape (K, D).

    timeline:
        Shape (T,), seconds.

    Returns
    -------
    np.ndarray
        Shape (T, D).
    """
    observation_times = np.asarray(
        observation_times,
        dtype=np.float64,
    )

    poses = np.asarray(
        poses,
        dtype=np.float64,
    )

    timeline = np.asarray(
        timeline,
        dtype=np.float64,
    )

    if poses.ndim != 2:
        raise ValueError(
            "poses must have shape (K, D)"
        )

    if observation_times.shape[0] != poses.shape[0]:
        raise ValueError(
            "observation_times and poses must have equal length"
        )

    aligned = np.full(
        (timeline.shape[0], poses.shape[1]),
        np.nan,
        dtype=np.float64,
    )

    for time, pose in zip(
        observation_times,
        poses,
    ):
        matches = np.flatnonzero(
            np.isclose(
                timeline,
                time,
                atol=atol,
                rtol=0.0,
            )
        )

        if matches.size == 0:
            raise ValueError(
                f"observation time {time} is not in timeline"
            )

        aligned[matches[0]] = pose

    return aligned

# Build six dimensional state
# [x,y,hx,hy] + [speed] + [hdot] --> [x, y, hx, hy, s, hdot]
def build_kinematic_state(
    poses: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an aligned pose trajectory into STRIVE state.

    Input pose:
        [x, y, hx, hy]

    Output state:
        [x, y, hx, hy, speed, heading_rate]

    Returns
    -------
    state:
        Shape (T, 6)

    visibility:
        Shape (T,), where 1 means the complete kinematic
        state is available.
    """
    poses = np.asarray(
        poses,
        dtype=np.float64,
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError(
            "poses must have shape (T, 4)"
        )

    if timestamps.shape != (poses.shape[0],):
        raise ValueError(
            "timestamps must have shape (T,)"
        )

    positions = poses[:, :2]

    velocities = velocity(
        positions,
        timestamps,
    )

    speed = np.linalg.norm(
        velocities,
        axis=1,
    )

    headings = np.arctan2(
        poses[:, 3],
        poses[:, 2],
    )

    heading_rate = heading_change_rate(
        headings,
        timestamps,
    )

    state = np.concatenate(
        [
            poses,
            speed[:, None],
            heading_rate[:, None],
        ],
        axis=1,
    )

    visibility = (
        ~np.isnan(speed)
        & ~np.isnan(heading_rate)
        & ~np.isnan(poses).any(axis=1)
    )

    # Match STRIVE's handling of isolated observations:
    # a pose without enough neighboring information to determine
    # motion is not considered a usable state.
    state[~visibility] = np.nan

    return (
        state,
        visibility.astype(np.float32),
    )


