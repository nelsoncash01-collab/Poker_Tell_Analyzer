import json

import pytest

from poker_tell.hand_history import Action, ActionType, HandHistory, Street
from poker_tell.hand_ingest import (
    HandHistoryStore,
    ingest_hand_history,
    label_source_summary,
    load_hands,
    load_hands_csv,
    load_hands_json,
    validate_hands,
)
from poker_tell.sync import SyncTable
from poker_tell.video import FrameClock

EXAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples"


# --- loaders ---------------------------------------------------------------

def test_json_and_csv_loaders_agree():
    from_json = load_hands_json(EXAMPLES / "hands.json")
    from_csv = load_hands_csv(EXAMPLES / "hands.csv")
    assert [h.hand_id for h in from_json] == [h.hand_id for h in from_csv]
    # Compare the first hand in full (actions, hole cards, times).
    assert from_json[0] == from_csv[0]


def test_load_dispatch_by_extension():
    assert len(load_hands(EXAMPLES / "hands.json")) == 2
    assert len(load_hands(EXAMPLES / "hands.csv")) == 2
    with pytest.raises(ValueError):
        load_hands(EXAMPLES / "hands.json", fmt="bogus")


def test_csv_groups_actions_per_hand():
    hands = load_hands_csv(EXAMPLES / "hands.csv")
    h7 = next(h for h in hands if h.hand_id == "hsp_s1_e1_h7")
    assert len(h7.actions) == 4
    assert h7.actions[0].action_type == ActionType.RAISE
    assert h7.hole_cards == ("Ah", "Kd")
    assert h7.hole_cards_revealed is True


def test_csv_inconsistent_hand_level_field_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text(
        "hand_id,player_id,hand_start_time,street,actor,action_type\n"
        "h1,negreanu,10.0,PREFLOP,negreanu,bet\n"
        "h1,negreanu,99.0,FLOP,negreanu,bet\n"  # hand_start_time differs
    )
    with pytest.raises(ValueError):
        load_hands_csv(csv)


# --- validation ------------------------------------------------------------

def test_validate_clean_hands_no_errors():
    hands = load_hands_json(EXAMPLES / "hands.json")
    issues = validate_hands(hands, "negreanu")
    assert [i for i in issues if i.severity == "error"] == []


def test_validate_flags_foreign_player_and_duplicates():
    hands = [
        HandHistory("h1", "ivey", 1.0),
        HandHistory("h1", "negreanu", 2.0),
    ]
    issues = validate_hands(hands, "negreanu")
    codes = {(i.code, i.severity) for i in issues}
    assert ("foreign_player", "error") in codes
    assert ("duplicate_hand_id", "error") in codes


def test_validate_warns_on_backwards_street():
    h = HandHistory("h1", "negreanu", 1.0, [
        Action(Street.RIVER, "negreanu", ActionType.BET),
        Action(Street.FLOP, "negreanu", ActionType.BET),
    ])
    codes = {i.code for i in validate_hands([h], "negreanu")}
    assert "street_order" in codes


def test_validate_warns_on_hole_card_inconsistency():
    revealed_no_cards = HandHistory("h1", "negreanu", 1.0,
                                    hole_cards_revealed=True)
    cards_no_reveal = HandHistory("h2", "negreanu", 1.0,
                                  hole_cards=("Ah", "Kd"))
    codes = {i.code for i in validate_hands(
        [revealed_no_cards, cards_no_reveal], "negreanu")}
    assert "revealed_without_cards" in codes
    assert "cards_without_reveal" in codes


def test_label_source_summary():
    hands = [
        HandHistory("h1", "p", 1.0, hole_cards=("Ah", "Kd"),
                    hole_cards_revealed=True),
        HandHistory("h2", "p", 2.0),
    ]
    summ = label_source_summary(hands)
    assert summ == {
        "n_hands": 2, "n_revealed_hole_cards": 1, "n_unrevealed": 1,
        "revealed_fraction": 0.5,
    }


# --- store + ingest --------------------------------------------------------

def test_store_save_load_round_trip(tmp_path):
    store = HandHistoryStore.load(tmp_path, "negreanu")
    for h in load_hands_json(EXAMPLES / "hands.json"):
        store.add(h)
    store.save()

    reloaded = HandHistoryStore.load(tmp_path, "negreanu")
    assert len(reloaded) == 2
    assert "hsp_s1_e1_h7" in reloaded


def test_store_rejects_foreign_player(tmp_path):
    store = HandHistoryStore.load(tmp_path, "negreanu")
    with pytest.raises(PermissionError):
        store.add(HandHistory("h1", "ivey", 1.0))


def test_store_load_rejects_foreign_file(tmp_path):
    # Write a hands.json under negreanu's dir but with the wrong header.
    store = HandHistoryStore("negreanu", tmp_path)
    store.dir.mkdir(parents=True)
    store.hands_path.write_text(json.dumps({"player_id": "ivey", "hands": []}))
    with pytest.raises(PermissionError):
        HandHistoryStore.load(tmp_path, "negreanu")


def test_ingest_hand_history_writes_store(tmp_path):
    store, issues = ingest_hand_history(tmp_path, "negreanu",
                                        EXAMPLES / "hands.json")
    assert len(store) == 2
    assert (tmp_path / "data" / "negreanu" / "hand_history" / "hands.json").exists()
    assert [i for i in issues if i.severity == "error"] == []


def test_ingest_refuses_on_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([
        {"hand_id": "h1", "player_id": "ivey", "hand_start_time": 1.0,
         "actions": []},
    ]))
    with pytest.raises(ValueError):
        ingest_hand_history(tmp_path, "negreanu", bad)


def test_ingested_hands_feed_sync_coverage(tmp_path):
    """The seam this stage exists to fill: loaded hands drive SyncTable."""
    store, _ = ingest_hand_history(tmp_path, "negreanu", EXAMPLES / "hands.json")
    table = SyncTable("negreanu", FrameClock(30.0))
    cov = table.coverage(store.hands())
    # Nothing synced yet, so every hand is missing a frame mapping.
    assert set(cov["missing_sync"]) == {"hsp_s1_e1_h7", "hsp_s1_e1_h12"}
    assert cov["synced"] == []
