# category mapping
import numpy as np
import torch


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

