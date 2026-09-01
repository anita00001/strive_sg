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


# Generate world-space coordinates aligned with each agent
def gen_car_coords(
    positions: torch.Tensor,
    headings: torch.Tensor,
    num_channels: int,
    length_pixels: int,
    width_pixels: int,
    *,
    bounds: list[float] | tuple[float, float, float, float] | None = None,
    lengths: torch.Tensor | None = None,
    widths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate world-space sampling coordinates aligned with each agent.

    Parameters
    ----------
    positions:
        Shape (B, 2), world-space [x, y].

    headings:
        Shape (B, 2), unit heading vectors [hx, hy].

    num_channels:
        Number of map channels.

    length_pixels:
        Number of samples along the vehicle longitudinal axis.

    width_pixels:
        Number of samples along the vehicle lateral axis.

    bounds:
        [low_l, low_w, high_l, high_w] in meters.

    lengths, widths:
        Optional per-agent physical dimensions. Used instead of
        bounds when sampling the area occupied by a vehicle.

    Returns
    -------
    torch.Tensor
        Shape:

            (B, C, L, W, 2)

        containing world-space [x, y] sample positions.
    """
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError(
            "positions must have shape (B, 2)"
        )

    if headings.ndim != 2 or headings.shape[-1] != 2:
        raise ValueError(
            "headings must have shape (B, 2)"
        )

    if positions.shape[0] != headings.shape[0]:
        raise ValueError(
            "positions and headings must have equal batch size"
        )

    batch_size = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    if bounds is not None:
        if len(bounds) != 4:
            raise ValueError(
                "bounds must contain four values"
            )

        longitudinal = torch.linspace(
            bounds[0],
            bounds[2],
            length_pixels,
            device=device,
            dtype=dtype,
        )

        lateral = torch.linspace(
            bounds[1],
            bounds[3],
            width_pixels,
            device=device,
            dtype=dtype,
        )

        longitudinal = longitudinal.view(
            1,
            1,
            length_pixels,
            1,
        ).expand(
            batch_size,
            num_channels,
            length_pixels,
            width_pixels,
        )

        lateral = lateral.view(
            1,
            1,
            1,
            width_pixels,
        ).expand(
            batch_size,
            num_channels,
            length_pixels,
            width_pixels,
        )

    elif lengths is not None and widths is not None:

        longitudinal = torch.linspace(
            -1.0,
            1.0,
            length_pixels,
            device=device,
            dtype=dtype,
        )

        lateral = torch.linspace(
            -1.0,
            1.0,
            width_pixels,
            device=device,
            dtype=dtype,
        )

        longitudinal = longitudinal.view(
            1,
            1,
            length_pixels,
            1,
        ).expand(
            batch_size,
            num_channels,
            length_pixels,
            width_pixels,
        )

        lateral = lateral.view(
            1,
            1,
            1,
            width_pixels,
        ).expand(
            batch_size,
            num_channels,
            length_pixels,
            width_pixels,
        )

        longitudinal = (
            longitudinal
            * lengths.view(batch_size, 1, 1, 1)
            / 2.0
        )

        lateral = (
            lateral
            * widths.view(batch_size, 1, 1, 1)
            / 2.0
        )

    else:
        raise ValueError(
            "provide either bounds or both lengths and widths"
        )

    hx = headings[:, 0].view(
        batch_size,
        1,
        1,
        1,
    )

    hy = headings[:, 1].view(
        batch_size,
        1,
        1,
        1,
    )

    # Rotate local coordinates into world coordinates.
    world_x = (
        longitudinal * hx
        - lateral * hy
    )

    world_y = (
        longitudinal * hy
        + lateral * hx
    )

    world_coordinates = torch.stack(
        [
            world_x,
            world_y,
        ],
        dim=-1,
    )

    world_coordinates = (
        world_coordinates
        + positions.view(
            batch_size,
            1,
            1,
            1,
            2,
        )
    )

    return world_coordinates

# sample an agent-oriented crop from a raster map
def get_map_obs(
    maps: torch.Tensor,
    meters_per_pixel: torch.Tensor,
    frames: torch.Tensor,
    map_indices: torch.Tensor,
    bounds: list[float] | tuple[float, float, float, float],
    *,
    length_pixels: int = 256,
    width_pixels: int = 256,
) -> torch.Tensor:
    """
    Sample agent-oriented local map observations.

    Parameters
    ----------
    maps:
        Rasterized maps with shape:

            (M, C, H, W)

        where M is number of maps.

    meters_per_pixel:
        Shape (M, 2), containing [dy, dx]-compatible
        map resolution values in meters/pixel.

    frames:
        Shape (B, 4):

            [x, y, hx, hy]

    map_indices:
        Shape (B,), selecting which global map each frame uses.

    bounds:
        [low_l, low_w, high_l, high_w] in meters.

    Returns
    -------
    torch.Tensor
        Local raster observations:

            (B, C, L, W)
    """
    if maps.ndim != 4:
        raise ValueError(
            "maps must have shape (M, C, H, W)"
        )

    if frames.ndim != 2 or frames.shape[-1] != 4:
        raise ValueError(
            "frames must have shape (B, 4)"
        )

    if map_indices.ndim != 1:
        raise ValueError(
            "map_indices must have shape (B,)"
        )

    batch_size = frames.shape[0]

    if map_indices.shape[0] != batch_size:
        raise ValueError(
            "frames and map_indices must have equal batch size"
        )

    num_channels = maps.shape[1]

    coordinates = gen_car_coords(
        positions=frames[:, :2],
        headings=frames[:, 2:4],
        num_channels=num_channels,
        length_pixels=length_pixels,
        width_pixels=width_pixels,
        bounds=bounds,
    )

    # Avoid invalid integer conversion for NaN poses.
    coordinates = torch.nan_to_num(
        coordinates,
        nan=0.0,
    )

    resolution = meters_per_pixel[
        map_indices
    ].view(
        batch_size,
        1,
        1,
        1,
        2,
    )

    pixel_coordinates = torch.round(
        coordinates / resolution
    ).long()

    x_pixels = pixel_coordinates[..., 0]
    y_pixels = pixel_coordinates[..., 1]

    map_height = maps.shape[2]
    map_width = maps.shape[3]

    outside = (
        (x_pixels < 0)
        | (x_pixels >= map_width)
        | (y_pixels < 0)
        | (y_pixels >= map_height)
    )

    # Match STRIVE's reference behavior:
    # out-of-map samples are redirected to pixel (0, 0).
    x_pixels = x_pixels.clone()
    y_pixels = y_pixels.clone()

    x_pixels[outside] = 0
    y_pixels[outside] = 0

    selected_maps = map_indices.view(
        batch_size,
        1,
        1,
        1,
    ).expand(
        batch_size,
        num_channels,
        length_pixels,
        width_pixels,
    )

    channel_indices = torch.arange(
        num_channels,
        device=maps.device,
    ).view(
        1,
        num_channels,
        1,
        1,
    ).expand(
        batch_size,
        num_channels,
        length_pixels,
        width_pixels,
    )

    return maps[
        selected_maps,
        channel_indices,
        y_pixels,
        x_pixels,
    ]


