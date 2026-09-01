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

