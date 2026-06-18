"""Per-player namespaced storage paths, with a cross-player leakage guard.

The single most important invariant in this codebase: nothing learned about one
player may leak into another player's model or normalization stats. Storage is
namespaced ``data/<player_id>/...`` and ``models/<player_id>/...``. The helpers
here make that layout the path of least resistance, and ``guard_single_player``
turns an accidental cross-namespace read into a loud error instead of silent
contamination.
"""

from __future__ import annotations

import re
from pathlib import Path

# Conservative player_id charset: keeps ids usable as directory names across
# platforms and makes namespace parsing unambiguous.
_PLAYER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_player_id(player_id: str) -> str:
    """Return ``player_id`` if it is a safe directory-name token, else raise."""
    if not isinstance(player_id, str) or not _PLAYER_ID_RE.match(player_id):
        raise ValueError(
            f"invalid player_id {player_id!r}; expected token matching "
            f"{_PLAYER_ID_RE.pattern}"
        )
    return player_id


def player_data_dir(root: Path | str, player_id: str) -> Path:
    """``<root>/data/<player_id>`` — raw footage, hand history, feature tables."""
    return Path(root) / "data" / validate_player_id(player_id)


def player_models_dir(root: Path | str, player_id: str) -> Path:
    """``<root>/models/<player_id>`` — trained, per-player model artifacts."""
    return Path(root) / "models" / validate_player_id(player_id)


def player_id_of_path(root: Path | str, path: Path | str) -> str | None:
    """Infer the owning player_id of ``path`` under ``data/`` or ``models/``.

    Returns ``None`` if ``path`` is not inside a recognized player namespace.
    """
    root = Path(root).resolve()
    p = Path(path).resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) >= 2 and parts[0] in ("data", "models"):
        return parts[1]
    return None


def guard_single_player(
    player_id: str, paths: list[Path | str], root: Path | str
) -> None:
    """Raise if any of ``paths`` belongs to a different player's namespace.

    Call this at the boundary of any per-player operation (loading features,
    fitting a model, computing normalization stats) to make cross-player
    contamination fail loudly rather than silently degrade the no-transfer
    guarantee. Paths outside any player namespace (shared code, configs) are
    ignored.
    """
    validate_player_id(player_id)
    offenders = []
    for path in paths:
        owner = player_id_of_path(root, path)
        if owner is not None and owner != player_id:
            offenders.append((str(path), owner))
    if offenders:
        details = "; ".join(f"{p} (owned by {o})" for p, o in offenders)
        raise PermissionError(
            f"cross-player access blocked for player {player_id!r}: {details}. "
            "Per-individual models must never read another player's data."
        )
