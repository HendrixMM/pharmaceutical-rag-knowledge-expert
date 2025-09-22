import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def load_dotenv_if_available():
    """Load environment variables from .env if python-dotenv is installed.

    Keeps tests flexible for local runs without committing secrets.
    """
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except Exception:
        return

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
        # Also export to os.environ for subprocesses
        # load_dotenv already injects into os.environ by default.
        # No extra work needed here.

