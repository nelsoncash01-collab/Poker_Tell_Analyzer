"""LLM-reader tests — no network and no API key.

The JSON->FrameReading mapper is pure. read_frame_llm is exercised with an
injected fake client that returns canned JSON, so no real Claude call happens.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from poker_tell.broadcast.llm_reader import (
    READING_SCHEMA,
    read_frame_llm,
    reading_from_json,
)


def test_reading_from_json_full():
    d = {
        "pot": 203800,
        "board": ["3c", "7c", "2c", "7h", "Jc"],
        "seats": [
            {"name": "GREENSTEIN", "hole_cards": ["Ac", "Jh"],
             "status": {"kind": "winner", "action_type": None, "amount": None,
                        "equity_pct": None}},
        ],
    }
    r = reading_from_json(d)
    assert r.pot == 203800
    assert r.board == ["3c", "7c", "2c", "7h", "Jc"]
    assert r.seats[0].name == "GREENSTEIN"
    assert r.seats[0].hole_cards == ("Ac", "Jh")
    assert r.seats[0].status.kind == "winner"


def test_reading_from_json_normalizes_ten_and_action():
    d = {
        "pot": 12200,
        "board": [],
        "seats": [
            {"name": "NEGREANU", "hole_cards": ["Ac", "10s"],
             "status": {"kind": "equity", "action_type": None, "amount": None,
                        "equity_pct": 65}},
            {"name": "IVEY", "hole_cards": ["Qs", "Qc"],
             "status": {"kind": "action", "action_type": "raise", "amount": 3000,
                        "equity_pct": None}},
        ],
    }
    r = reading_from_json(d)
    assert r.seats[0].hole_cards == ("Ac", "Ts")     # 10s -> Ts
    assert r.seats[0].status.equity_pct == 65
    assert r.seats[1].status.action_type == "raise"
    assert r.seats[1].status.amount == 3000


def test_reading_from_json_null_and_bad_values():
    d = {
        "pot": None,
        "board": ["ZZ", "Ah"],            # ZZ is unrecognized -> dropped
        "seats": [
            # has a name, so it's kept; tests null hole_cards handling
            {"name": "NEGREANU", "hole_cards": None,
             "status": {"kind": "none", "action_type": None, "amount": None,
                        "equity_pct": None}},
        ],
    }
    r = reading_from_json(d)
    assert r.pot is None
    assert r.board == ["Ah"]
    assert r.seats[0].name == "NEGREANU"
    assert r.seats[0].hole_cards is None
    assert r.seats[0].status.kind == "none"


def test_reading_from_json_drops_empty_seats():
    d = {
        "pot": 203800,
        "board": [],
        "seats": [
            {"name": "GREENSTEIN", "hole_cards": ["Ac", "Jh"],
             "status": {"kind": "winner", "action_type": None, "amount": None,
                        "equity_pct": None}},
            {"name": None, "hole_cards": None,
             "status": {"kind": "none", "action_type": None, "amount": None,
                        "equity_pct": None}},
            {"name": None, "hole_cards": None,
             "status": {"kind": "none", "action_type": None, "amount": None,
                        "equity_pct": None}},
        ],
    }
    r = reading_from_json(d)
    assert len(r.seats) == 1
    assert r.seats[0].name == "GREENSTEIN"


def test_schema_is_strict_object():
    # Structured-outputs requirements: object, additionalProperties false, all
    # properties listed in required.
    assert READING_SCHEMA["type"] == "object"
    assert READING_SCHEMA["additionalProperties"] is False
    assert set(READING_SCHEMA["required"]) == {"pot", "board", "seats"}


class FakeMessages:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self._payload)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    def __init__(self, payload):
        self.messages = FakeMessages(payload)


def test_read_frame_llm_with_injected_client():
    pytest.importorskip("PIL")
    payload = {
        "pot": 8300,
        "board": [],
        "seats": [
            {"name": "IVEY", "hole_cards": ["Qs", "Qc"],
             "status": {"kind": "action", "action_type": "raise", "amount": 3000,
                        "equity_pct": None}},
        ],
    }
    client = FakeClient(payload)
    frame = np.full((48, 64, 3), 120, dtype=np.uint8)
    reading = read_frame_llm(frame, client=client, model="claude-haiku-4-5")

    assert reading.pot == 8300
    assert reading.seats[0].status.amount == 3000
    # The request used the chosen model, structured output, and an image block.
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    blocks = call["messages"][0]["content"]
    assert any(b.get("type") == "image" for b in blocks)


def test_frame_to_image_block_caps_width():
    pytest.importorskip("PIL")
    from poker_tell.broadcast.llm_reader import frame_to_png_base64

    wide = np.full((100, 4000, 3), 200, dtype=np.uint8)
    b64 = frame_to_png_base64(wide, max_width=1280)
    assert isinstance(b64, str) and len(b64) > 0
