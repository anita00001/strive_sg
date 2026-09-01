import numpy as np

from src.datasets.map_env import (
    DEFAULT_MAP_LAYERS,
    build_layer_map,
    compute_raster_geometry,
    maybe_flip_singapore_raster,
    num_raster_channels,
    rasterize_map_layers,
)

from src.datasets.map_env import FakeNuScenesMap

# test default channels
def test_default_layer_map():
    result = build_layer_map(
        DEFAULT_MAP_LAYERS
    )

    assert result == {
        "drivable_area": 0,
        "carpark_area": 1,
        "road_divider": 2,
        "lane_divider": 3,
    }

    assert num_raster_channels(
        DEFAULT_MAP_LAYERS
    ) == 4

# Test road merging
def test_road_layers_share_channel():
    layers = [
        "drivable_area",
        "road_segment",
        "lane",
        "lane_divider",
    ]

    result = build_layer_map(
        layers
    )

    assert result == {
        "drivable_area": 0,
        "road_segment": 0,
        "lane": 0,
        "lane_divider": 1,
    }

    assert num_raster_channels(
        layers
    ) == 2

# test geometry
def test_compute_raster_geometry():
    raster_size, resolution = (
        compute_raster_geometry(
            map_size_meters=(
                100.0,
                50.0,
            ),
            pixels_per_meter=4.0,
        )
    )

    assert raster_size == (
        400,
        200,
    )

    assert np.allclose(
        resolution,
        np.array(
            [
                0.25,
                0.25,
            ]
        ),
    )

# Test Singapore flipping
def test_flip_singapore_raster():
    raster = np.array(
        [
            [
                [1, 2],
                [3, 4],
            ]
        ]
    )

    result = maybe_flip_singapore_raster(
        raster,
        "singapore-onenorth",
        flip_singapore=True,
    )

    expected = np.array(
        [
            [
                [3, 4],
                [1, 2],
            ]
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )

# Test Boston raster is not flipped
def test_does_not_flip_boston():
    raster = np.array(
        [
            [
                [1, 2],
                [3, 4],
            ]
        ]
    )

    result = maybe_flip_singapore_raster(
        raster,
        "boston-seaport",
        flip_singapore=True,
    )

    assert np.array_equal(
        result,
        raster,
    )

# Test rasterization without nuScenes data
def test_rasterize_map_layers():
    fake_map = FakeNuScenesMap()

    raster = rasterize_map_layers(
        fake_map,
        layers=[
            "drivable_area",
            "road_segment",
            "carpark_area",
            "lane_divider",
        ],
        map_size_pixels=(
            3,
            3,
        ),
    )

    assert raster.shape == (
        3,
        3,
        3,
    )

    # Road layers merged into channel 0.
    assert raster[0, 0, 0] == 1
    assert raster[0, 0, 1] == 1

    # carpark_area
    assert raster[1, 1, 0] == 1

    # lane_divider
    assert raster[2, 1, 1] == 1


