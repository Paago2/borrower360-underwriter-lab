from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    checks: Dict[str, Dict[str, str]]  # name -> {status, detail}


def _file_check(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.is_dir():
        return False, "is a directory"
    try:
        with path.open("rb") as f:
            f.read(64)
        return True, "ok"
    except Exception as e:
        return False, f"unreadable: {type(e).__name__}: {e}"


def _dir_check(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if not path.is_dir():
        return False, "not a directory"
    try:
        _ = next(path.iterdir(), None)
        return True, "ok"
    except Exception as e:
        return False, f"unreadable: {type(e).__name__}: {e}"


def _any_path_exists(*paths: Path) -> Tuple[bool, str]:
    for p in paths:
        if p.exists():
            return True, f"found: {p.as_posix()}"
    return False, "none found"


def check_readiness(
    configs_dir: Path = Path("/app/configs"),
    data_dir: Path = Path("/app/data"),
) -> ReadinessResult:
    checks: Dict[str, Dict[str, str]] = {}

    # Configs
    contracts_ok, contracts_detail = _file_check(configs_dir / "contracts.yaml")
    datasets_ok, datasets_detail = _file_check(configs_dir / "datasets.yaml")
    checks["configs.contracts"] = {"status": "ok" if contracts_ok else "fail", "detail": contracts_detail}
    checks["configs.datasets"] = {"status": "ok" if datasets_ok else "fail", "detail": datasets_detail}

    # Data mount
    data_ok, data_detail = _dir_check(data_dir)
    checks["data.mount"] = {"status": "ok" if data_ok else "fail", "detail": data_detail}

    # Minimum compliance artifact check (align to your structure: /app/data/00_raw/...)
    # We'll accept "any of these exist" to keep it robust while you're iterating.
    sanctions_ok, sanctions_detail = _any_path_exists(
        data_dir / "00_raw" / "compliance_sanctions",
        data_dir / "00_raw" / "sanctions",
        data_dir / "00_raw" / "ofac",
    )
    checks["sanctions.input_dir"] = {"status": "ok" if sanctions_ok else "fail", "detail": sanctions_detail}

    ready = all(v["status"] == "ok" for v in checks.values())
    return ReadinessResult(ready=ready, checks=checks)
