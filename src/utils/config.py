import configargparse


def get_parser(description: str) -> configargparse.ArgParser:
    """
    Create a parser that supports both command-line arguments
    and YAML configuration files.
    """
    parser = configargparse.ArgParser(
        description=description,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add_argument(
        "-c",
        "--config",
        required=False,
        is_config_file=True,
        help="Path to YAML configuration file.",
    )

    return parser


def add_base_args(
    parser: configargparse.ArgParser,
) -> configargparse.ArgParser:
    """
    Add arguments shared by training, testing, and optimization scripts.
    """

    # Output
    parser.add_argument(
        "--out",
        type=str,
        default="./out/traffic_out",
        help="Directory for logs, checkpoints, and results.",
    )

    # Dataset
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/nuscenes",
        help="Root directory containing nuScenes data.",
    )

    parser.add_argument(
        "--data_version",
        type=str,
        choices=["trainval", "mini"],
        default="trainval",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    # Trajectory
    parser.add_argument(
        "--past_len",
        type=int,
        default=4,
        help="Number of observed past timesteps.",
    )

    parser.add_argument(
        "--future_len",
        type=int,
        default=12,
        help="Number of future timesteps to predict.",
    )

    parser.add_argument(
        "--agent_types",
        type=str,
        nargs="+",
        default=["car", "truck"],
    )

    # Map
    parser.add_argument(
        "--map_obs_size_pix",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--map_obs_bounds",
        type=float,
        nargs=4,
        default=[-17.0, -38.5, 60.0, 38.5],
    )

    parser.add_argument(
        "--map_layers",
        type=str,
        nargs="+",
        default=[
            "drivable_area",
            "carpark_area",
            "road_divider",
            "lane_divider",
        ],
    )

    # Traffic model
    parser.add_argument(
        "--map_feat_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--past_feat_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--future_feat_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--latent_size",
        type=int,
        default=32,
    )

    return parser