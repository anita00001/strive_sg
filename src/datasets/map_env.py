from __future__ import annotations

import numpy as np
import torch

from pathlib import Path

from nuscenes.map_expansion.map_api import NuScenesMap

from src.datasets.nuscenes_utils import get_map_obs


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

NUSC_MAP_NAMES = (
    "singapore-hollandvillage",
    "singapore-queenstown",
    "boston-seaport",
    "singapore-onenorth",
)


def load_nuscenes_maps(
    data_path: str | Path,
) -> dict[str, NuScenesMap]:
    """
    Load the four nuScenes HD maps.
    """
    return {
        map_name: NuScenesMap(
            dataroot=str(data_path),
            map_name=map_name,
        )
        for map_name in NUSC_MAP_NAMES
    }

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

# nuScenesMapEnv
class NuScenesMapEnv:
    """
    Rasterized nuScenes map environment used by STRIVE.

    The full semantic maps are rasterized once and cached.
    Local agent-oriented crops can then be sampled efficiently.
    """

    def __init__(
        self,
        map_data_path: str | Path | None = None,
        *,
        bounds: tuple[float, float, float, float] = (
            -17.0,
            -38.5,
            60.0,
            38.5,
        ),
        layers: tuple[str, ...] = DEFAULT_MAP_LAYERS,
        length_pixels: int = 256,
        width_pixels: int = 256,
        device: str | torch.device = "cpu",
        flip_singapore: bool = True,
        pixels_per_meter: float = 4.0,
        nusc_maps: dict | None = None,
        map_sizes: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        map_data_path:
            nuScenes dataset root.

        bounds:
            Local crop bounds:
                [behind, lateral_low, front, lateral_high]

        layers:
            Semantic nuScenes layers to rasterize.

        length_pixels, width_pixels:
            Resolution of local map crops.

        device:
            Device where full raster maps are cached.

        flip_singapore:
            Reflect Singapore maps to match STRIVE's convention.

        pixels_per_meter:
            Full-map rasterization resolution.

        nusc_maps:
            Optional pre-loaded map objects.
            Useful for testing.

        map_sizes:
            Optional map-size dictionary.
            Defaults to official nuScenes map dimensions.
        """
        self.data_path = (
            Path(map_data_path)
            if map_data_path is not None
            else None
        )

        self.bounds = tuple(bounds)
        self.layer_names = tuple(layers)

        self.L = length_pixels
        self.W = width_pixels

        self.device = torch.device(device)

        self.flip_singapore = flip_singapore
        self.pixels_per_meter = pixels_per_meter

        self.map_sizes = (
            NUSC_MAP_SIZES
            if map_sizes is None
            else map_sizes
        )

        if nusc_maps is None:
            if self.data_path is None:
                raise ValueError(
                    "map_data_path is required when "
                    "nusc_maps is not supplied"
                )

            nusc_maps = load_nuscenes_maps(
                self.data_path
            )

        self.nusc_maps = nusc_maps
        self.map_list = list(
            self.nusc_maps.keys()
        )

        self.layer_map = build_layer_map(
            self.layer_names
        )

        self.num_layers = num_raster_channels(
            self.layer_names
        )

        self._build_raster_cache()

    def _build_raster_cache(self) -> None:
        """
        Rasterize every map and pad them to one common tensor size.
        """
        rasters = []
        resolutions = []

        max_height = 0
        max_width = 0

        raster_sizes = {}

        # First determine raster size for every map.
        for map_name in self.map_list:
            if map_name not in self.map_sizes:
                raise ValueError(
                    f"unknown map size for {map_name}"
                )

            raster_size, resolution = (
                compute_raster_geometry(
                    self.map_sizes[map_name],
                    self.pixels_per_meter,
                )
            )

            raster_sizes[map_name] = (
                raster_size
            )

            resolutions.append(
                resolution
            )

            max_height = max(
                max_height,
                raster_size[0],
            )

            max_width = max(
                max_width,
                raster_size[1],
            )

        # Rasterize each semantic map.
        for map_name in self.map_list:
            raster_size = raster_sizes[
                map_name
            ]

            raster = rasterize_map_layers(
                self.nusc_maps[map_name],
                self.layer_names,
                raster_size,
            )

            raster = (
                maybe_flip_singapore_raster(
                    raster,
                    map_name,
                    flip_singapore=(
                        self.flip_singapore
                    ),
                )
            )

            height = raster.shape[1]
            width = raster.shape[2]

            pad_height = (
                max_height - height
            )

            pad_width = (
                max_width - width
            )

            padded = np.pad(
                raster,
                (
                    (0, 0),
                    (0, pad_height),
                    (0, pad_width),
                ),
                mode="constant",
                constant_values=0,
            )

            rasters.append(
                torch.from_numpy(
                    padded
                )
            )

        self.nusc_raster = torch.stack(
            rasters,
            dim=0,
        ).to(
            self.device
        )

        self.nusc_dx = torch.tensor(
            np.stack(
                resolutions,
                axis=0,
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def get_map_crop_pos(
        self,
        pos: torch.Tensor,
        map_indices: torch.Tensor,
        *,
        bounds=None,
        length_pixels: int | None = None,
        width_pixels: int | None = None,
    ) -> torch.Tensor:
        """
        Produce agent-oriented map crops.

        pos:
            Shape (N, 4):
                [x, y, hx, hy]

        map_indices:
            Shape (N,)
        """
        bounds = (
            self.bounds
            if bounds is None
            else bounds
        )

        length_pixels = (
            self.L
            if length_pixels is None
            else length_pixels
        )

        width_pixels = (
            self.W
            if width_pixels is None
            else width_pixels
        )

        output_device = pos.device

        map_obs = get_map_obs(
            maps=self.nusc_raster,
            meters_per_pixel=self.nusc_dx,
            frames=pos.to(
                self.device
            ),
            map_indices=map_indices.to(
                self.device
            ),
            bounds=bounds,
            length_pixels=length_pixels,
            width_pixels=width_pixels,
        )

        return map_obs.to(
            output_device
        )

    def get_map_crop(
        self,
        scene_graph,
        map_idx: torch.Tensor,
        *,
        bounds=None,
        length_pixels: int | None = None,
        width_pixels: int | None = None,
    ) -> torch.Tensor:
        """
        Produce map crops for every agent in a batched scene graph.

        scene_graph.pos:
            (N, 4) or (N, S, 4)

        scene_graph.batch:
            identifies which scene each agent belongs to.

        map_idx:
            shape (B,), one map index per scene.
        """
        if not hasattr(
            scene_graph,
            "pos",
        ):
            raise ValueError(
                "scene_graph must contain pos"
            )

        if not hasattr(
            scene_graph,
            "batch",
        ):
            raise ValueError(
                "scene_graph must contain batch"
            )

        map_indices = map_idx[
            scene_graph.batch
        ]

        positions = scene_graph.pos

        # Sometimes STRIVE asks for map crops at several
        # positions per agent.
        if positions.ndim == 3:
            num_agents = positions.shape[0]
            num_samples = positions.shape[1]

            positions = positions.reshape(
                num_agents * num_samples,
                4,
            )

            map_indices = (
                map_indices
                .unsqueeze(1)
                .expand(
                    num_agents,
                    num_samples,
                )
                .reshape(-1)
            )

        return self.get_map_crop_pos(
            positions,
            map_indices,
            bounds=bounds,
            length_pixels=length_pixels,
            width_pixels=width_pixels,
        )

class FullFakeMap:
    def get_map_mask(
        self,
        patch_box,
        patch_angle,
        layer_names,
        canvas_size,
    ):
        height, width = canvas_size

        masks = []

        for index, _ in enumerate(
            layer_names
        ):
            mask = np.ones(
                (height, width),
                dtype=np.uint8,
            )

            masks.append(mask)

        return np.stack(
            masks,
            axis=0,
        )

    