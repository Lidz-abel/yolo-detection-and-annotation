from __future__ import annotations

from pathlib import Path


def _parse_value(raw: str):
    value = raw.strip()
    if not value:
        raise ValueError("Empty TOML value is not supported.")

    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]

    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        if "." in value or "e" in lower:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_toml(path: str | Path) -> dict:
    """Load a small subset of TOML used by this project."""
    path = Path(path)
    config: dict[str, dict] = {}
    current_section: dict | None = None

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            if not section_name:
                raise ValueError(f"Invalid empty TOML section at line {lineno}.")
            current_section = config.setdefault(section_name, {})
            continue

        if "=" not in line:
            raise ValueError(f"Unsupported TOML line {lineno}: {raw_line}")

        if current_section is None:
            raise ValueError(f"TOML key must be inside a section at line {lineno}.")

        key, value = line.split("=", 1)
        current_section[key.strip()] = _parse_value(value)

    return config
