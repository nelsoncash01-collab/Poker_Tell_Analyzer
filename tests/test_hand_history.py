from poker_tell.hand_history import Action, ActionType, HandHistory, Street


def test_action_dict_round_trip():
    a = Action(Street.FLOP, "negreanu", ActionType.BET, 4000.0, wall_clock=1290.0)
    a2 = Action.from_dict(a.to_dict())
    assert a2 == a
    assert a.to_dict()["street"] == "FLOP"
    assert a.to_dict()["action_type"] == "bet"


def test_action_optional_wall_clock():
    a = Action(Street.PREFLOP, "x", ActionType.CHECK)
    d = a.to_dict()
    assert d["wall_clock"] is None
    assert Action.from_dict(d) == a


def test_hand_dict_round_trip_with_hole_cards():
    h = HandHistory(
        hand_id="h1", player_id="negreanu", hand_start_time=1234.5,
        actions=[
            Action(Street.PREFLOP, "negreanu", ActionType.RAISE, 2500, 1235.0),
            Action(Street.FLOP, "negreanu", ActionType.BET, 4000, 1290.0),
        ],
        hole_cards=("Ah", "Kd"), hole_cards_revealed=True,
    )
    h2 = HandHistory.from_dict(h.to_dict())
    assert h2 == h
    assert isinstance(h2.hole_cards, tuple)


def test_hand_dict_round_trip_without_hole_cards():
    h = HandHistory("h2", "negreanu", 10.0)
    h2 = HandHistory.from_dict(h.to_dict())
    assert h2 == h
    assert h2.hole_cards is None
    assert h2.hole_cards_revealed is False


def test_from_dict_ignores_unknown_keys():
    d = {
        "hand_id": "h3", "player_id": "negreanu", "hand_start_time": 1.0,
        "actions": [], "extra_field": "ignored", "future_thing": 42,
    }
    h = HandHistory.from_dict(d)
    assert h.hand_id == "h3"
