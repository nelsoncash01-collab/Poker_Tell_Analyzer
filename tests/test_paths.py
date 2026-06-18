import pytest

from poker_tell.paths import (
    guard_single_player,
    player_data_dir,
    player_id_of_path,
    player_models_dir,
    validate_player_id,
)


def test_validate_player_id():
    assert validate_player_id("negreanu") == "negreanu"
    assert validate_player_id("player_01-A") == "player_01-A"
    for bad in ["", "../etc", "has space", "/abs", "a/b"]:
        with pytest.raises(ValueError):
            validate_player_id(bad)


def test_namespaced_dirs(tmp_path):
    d = player_data_dir(tmp_path, "negreanu")
    m = player_models_dir(tmp_path, "negreanu")
    assert d == tmp_path / "data" / "negreanu"
    assert m == tmp_path / "models" / "negreanu"


def test_player_id_of_path(tmp_path):
    p = player_data_dir(tmp_path, "negreanu") / "video" / "ep1.mp4"
    assert player_id_of_path(tmp_path, p) == "negreanu"
    # outside any namespace
    assert player_id_of_path(tmp_path, tmp_path / "README.md") is None


def test_guard_blocks_cross_player(tmp_path):
    mine = player_data_dir(tmp_path, "negreanu") / "features.parquet"
    theirs = player_data_dir(tmp_path, "ivey") / "features.parquet"
    shared = tmp_path / "config.yaml"

    # Own + shared paths are fine.
    guard_single_player("negreanu", [mine, shared], root=tmp_path)

    # Another player's path is blocked.
    with pytest.raises(PermissionError):
        guard_single_player("negreanu", [mine, theirs], root=tmp_path)
