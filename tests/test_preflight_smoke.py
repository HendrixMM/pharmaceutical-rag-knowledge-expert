import json
import subprocess
from pathlib import Path


def test_ai_preflight_non_strict(tmp_path: Path):
    out = tmp_path / "preflight.json"
    rc = subprocess.run(
        [
            "python",
            "scripts/ai_preflight.py",
            "--output",
            str(out),
            "--pretty",
        ],
        check=False,
    )
    assert out.exists(), "preflight JSON should be written"
    data = json.loads(out.read_text())
    assert isinstance(data.get("results"), list)


def test_ai_smoke_non_strict(tmp_path: Path):
    out = tmp_path / "smoke.json"
    rc = subprocess.run(
        [
            "python",
            "scripts/ai_smoke_test.py",
            "--output",
            str(out),
        ],
        check=False,
    )
    assert out.exists(), "smoke JSON should be written"
    data = json.loads(out.read_text())
    steps = data.get("steps")
    assert isinstance(steps, list) and len(steps) > 0
