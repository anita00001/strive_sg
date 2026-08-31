import pytest

from src.utils.logger import Logger, throw_error


def test_logger_prints_to_terminal(capsys):
    Logger.reset()

    Logger.log("hello STRIVE")

    captured = capsys.readouterr()

    assert "hello STRIVE" in captured.out


def test_logger_writes_to_file(tmp_path):
    log_file = tmp_path / "logs" / "train.log"

    Logger.init(log_file)
    Logger.log("training started")

    assert log_file.exists()

    contents = log_file.read_text(encoding="utf-8")

    assert "training started" in contents

    Logger.reset()


def test_throw_error_raises_runtime_error():
    Logger.reset()

    with pytest.raises(RuntimeError, match="something failed"):
        throw_error("something failed")