from __future__ import annotations

import numpy as np
import torch


NUSC_MAP_SIZES = {
    "singapore-onenorth": (2025.0, 1585.6),
    "singapore-hollandvillage": (2922.9, 2808.3),
    "singapore-queenstown": (3687.1, 3228.6),
    "boston-seaport": (2118.1, 2979.5),
}


ROAD_LAYERS = {
    "drivable_area",
    "road_segment",
    "lane",
}


DEFAULT_MAP_LAYERS = (
    "drivable_area",
    "carpark_area",
    "road_divider",
    "lane_divider",
)


def build_layer_map(
    layers: list[str] | tuple[str, ...],
) -> dict[str, int]:
    """
    Map nuScenes semantic layer names to raster channel indices.

    Road-type layers share channel 0. All other requested layers
    receive separate channels.
    """
    road_layers = [
        layer
        for layer in layers
        if layer in ROAD_LAYERS
    ]

    other_layers = [
        layer
        for layer in layers
        if layer not in ROAD_LAYERS
    ]

    layer_map: dict[str, int] = {}

    if road_layers:
        for layer in road_layers:
            layer_map[layer] = 0

        next_channel = 1
    else:
        next_channel = 0

    for layer in other_layers:
        layer_map[layer] = next_channel
        next_channel += 1

    return layer_map


def num_raster_channels(
    layers: list[str] | tuple[str, ...],
) -> int:
    """
    Return number of raster channels generated for the layer set.
    """
    if not layers:
        return 0

    layer_map = build_layer_map(layers)

    return max(layer_map.values()) + 1

# Full-map rasterization
def rasterize_map_layers(
    nusc_map,
    layers: list[str] | tuple[str, ...],
    map_size_pixels: tuple[int, int],
) -> np.ndarray:
    """
    Rasterize requested nuScenes semantic map layers.

    Parameters
    ----------
    nusc_map:
        Object exposing the nuScenes Map API method `get_map_mask`.

    layers:
        Requested semantic layer names.

    map_size_pixels:
        (height, width) of full raster.

    Returns
    -------
    np.ndarray
        Binary raster with shape:

            (C, H, W)
    """
    if not layers:
        raise ValueError(
            "at least one map layer must be provided"
        )

    road_layers = [
        layer
        for layer in layers
        if layer in ROAD_LAYERS
    ]

    other_layers = [
        layer
        for layer in layers
        if layer not in ROAD_LAYERS
    ]

    raster_channels = []

    if road_layers:
        road_mask = nusc_map.get_map_mask(
            patch_box=None,
            patch_angle=0.0,
            layer_names=road_layers,
            canvas_size=map_size_pixels,
        )

        road_mask = np.asarray(
            road_mask,
            dtype=np.uint8,
        )

        # Union all requested road layers.
        road_mask = np.clip(
            road_mask.sum(axis=0),
            0,
            1,
        )

        raster_channels.append(
            road_mask[None]
        )

    if other_layers:
        other_mask = nusc_map.get_map_mask(
            patch_box=None,
            patch_angle=0.0,
            layer_names=other_layers,
            canvas_size=map_size_pixels,
        )

        raster_channels.append(
            np.asarray(
                other_mask,
                dtype=np.uint8,
            )
        )

    return np.concatenate(
        raster_channels,
        axis=0,
    )

# Singapore flipping
def maybe_flip_singapore_raster(
    raster: np.ndarray,
    map_name: str,
    *,
    flip_singapore: bool,
) -> np.ndarray:
    """
    Flip Singapore raster maps about the x-axis when requested.
    """
    if (
        flip_singapore
        and map_name.startswith("singapore-")
    ):
        return np.flip(
            raster,
            axis=1,
        ).copy()

    return raster

# convert physical map size to raster resolution
def compute_raster_geometry(
    map_size_meters: tuple[float, float],
    pixels_per_meter: float,
) -> tuple[tuple[int, int], np.ndarray]:
    """
    Compute full raster dimensions and actual meters-per-pixel.

    Returns
    -------
    raster_size:
        (H, W)

    meters_per_pixel:
        [meters_per_pixel_y, meters_per_pixel_x]
    """
    if pixels_per_meter <= 0:
        raise ValueError(
            "pixels_per_meter must be positive"
        )

    map_size = np.asarray(
        map_size_meters,
        dtype=np.float64,
    )

    raster_size_array = np.round(
        map_size * pixels_per_meter
    ).astype(np.int64)

    meters_per_pixel = (
        map_size
        / raster_size_array
    )

    raster_size = (
        int(raster_size_array[0]),
        int(raster_size_array[1]),
    )

    return (
        raster_size,
        meters_per_pixel,
    )

# Rasterization without nuScenes data
class FakeNuScenesMap:
    def get_map_mask(
        self,
        patch_box,
        patch_angle,
        layer_names,
        canvas_size,
    ):
        height, width = canvas_size

        outputs = []

        for layer in layer_names:
            mask = np.zeros(
                (height, width),
                dtype=np.uint8,
            )

            if layer == "drivable_area":
                mask[0, 0] = 1

            elif layer == "road_segment":
                mask[0, 1] = 1

            elif layer == "carpark_area":
                mask[1, 0] = 1

            elif layer == "lane_divider":
                mask[1, 1] = 1

            outputs.append(mask)

        return np.stack(
            outputs,
            axis=0,
        )

