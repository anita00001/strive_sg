from src.utils.config import add_base_args, get_parser


def test_base_defaults():
    parser = get_parser("test")
    parser = add_base_args(parser)

    args = parser.parse_args([])

    assert args.batch_size == 4
    assert args.past_len == 4
    assert args.future_len == 12
    assert args.agent_types == ["car", "truck"]
    assert args.map_obs_size_pix == 256
    assert args.latent_size == 32


def test_command_line_override():
    parser = get_parser("test")
    parser = add_base_args(parser)

    args = parser.parse_args(
        [
            "--batch_size",
            "8",
            "--past_len",
            "6",
            "--latent_size",
            "64",
        ]
    )

    assert args.batch_size == 8
    assert args.past_len == 6
    assert args.latent_size == 64