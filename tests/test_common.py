from src.utils.common import dict_to_namespace, mkdir


def test_dict_to_namespace():
    cfg = dict_to_namespace(
        {
            "batch_size": 4,
            "learning_rate": 1e-5,
        }
    )

    assert cfg.batch_size == 4
    assert cfg.learning_rate == 1e-5


def test_mkdir(tmp_path):
    output_dir = tmp_path / "experiment" / "results"

    returned_path = mkdir(output_dir)

    assert returned_path == output_dir
    assert output_dir.exists()
    assert output_dir.is_dir()