from __future__ import annotations

from pathlib import Path
from typing import Optional

import questionary

ROOT = Path("clean_cache")


def is_cache_file(path: Path) -> bool:
    return path.name.endswith(".json.gz") or path.name.endswith(".json")


def choose_cache_file(root: Path = ROOT) -> Optional[Path]:
    root = root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    current = root

    while True:
        entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        choices = []

        if current != root:
            choices.append(questionary.Choice("..", value="__up__"))

        for entry in entries:
            if entry.is_dir():
                choices.append(questionary.Choice(f"[DIR] {entry.name}", value=entry))
            elif is_cache_file(entry):
                choices.append(questionary.Choice(entry.name, value=entry))

        choices.append(questionary.Choice("Cancel", value="__cancel__"))

        selected = questionary.select(
            f"Browsing: {current.relative_to(root.parent)}",
            choices=choices,
        ).ask()

        if selected is None or selected == "__cancel__":
            return None

        if selected == "__up__":
            current = current.parent
            continue

        if isinstance(selected, Path):
            if selected.is_dir():
                current = selected
            else:
                return selected
