from datetime import datetime
from pathlib import Path


class Logger:
    """
    Simple application-wide logger.

    Messages are always printed to the terminal. If a log file has been
    configured, messages are also appended to that file with a timestamp.
    """

    _log_file: Path | None = None

    @classmethod
    def init(cls, log_path: str | Path) -> None:
        log_path = Path(log_path)

        # Ensure the parent directory exists.
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cls._log_file = log_path

    @classmethod
    def log(cls, message: object) -> None:
        message = str(message)

        # Terminal output.
        print(message)

        # File logging is optional.
        if cls._log_file is None:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with cls._log_file.open("a", encoding="utf-8") as file:
            file.write(f"{timestamp}  {message}\n")

    @classmethod
    def reset(cls) -> None:
        """
        Disable file logging.

        Primarily useful for tests.
        """
        cls._log_file = None


def throw_error(message: str) -> None:
    """
    Log an error message and raise RuntimeError.
    """
    Logger.log(f"ERROR: {message}")
    raise RuntimeError(message)