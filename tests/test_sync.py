import pytest

from poker_tell.hand_history import (
    Action,
    ActionType,
    HandHistory,
    Street,
)
from poker_tell.sync import (
    HandSyncEntry,
    StreetBoundary,
    SyncTable,
)
from poker_tell.video import FrameClock


def make_entry(hand_id, start, end, *, video="ep1", hh_time=None, streets=None,
               player="negreanu"):
    return HandSyncEntry(
        hand_id=hand_id,
        player_id=player,
        video_id=video,
        start_frame=start,
        end_frame=end,
        streets=tuple(streets or ()),
        hh_start_time=hh_time,
    )


def streets_in_order(base):
    """Four well-formed, in-order, non-overlapping streets starting at base."""
    return [
        StreetBoundary(Street.PREFLOP, base + 0, base + 99),
        StreetBoundary(Street.FLOP, base + 100, base + 199),
        StreetBoundary(Street.TURN, base + 200, base + 299),
        StreetBoundary(Street.RIVER, base + 300, base + 399),
    ]


def test_clean_table_validates_without_errors():
    t = SyncTable("negreanu", FrameClock(30.0))
    t.add(make_entry("h1", 0, 399, streets=streets_in_order(0)))
    t.add(make_entry("h2", 500, 899, streets=streets_in_order(500)))
    issues = t.validate()
    assert [i for i in issues if i.severity == "error"] == []


def test_single_player_enforced_on_add():
    t = SyncTable("negreanu", FrameClock(30.0))
    with pytest.raises(PermissionError):
        t.add(make_entry("h1", 0, 399, player="ivey"))


def test_duplicate_hand_id_rejected():
    t = SyncTable("negreanu", FrameClock(30.0))
    t.add(make_entry("h1", 0, 399))
    with pytest.raises(ValueError):
        t.add(make_entry("h1", 400, 799))


def test_street_outside_hand_is_error():
    t = SyncTable("negreanu", FrameClock(30.0))
    bad = [StreetBoundary(Street.PREFLOP, 0, 500)]  # exceeds hand end 399
    t.add(make_entry("h1", 0, 399, streets=bad))
    codes = {i.code for i in t.validate() if i.severity == "error"}
    assert "street_outside_hand" in codes


def test_street_out_of_order_is_error():
    t = SyncTable("negreanu", FrameClock(30.0))
    bad = [
        StreetBoundary(Street.FLOP, 0, 99),
        StreetBoundary(Street.PREFLOP, 100, 199),
    ]
    t.add(make_entry("h1", 0, 399, streets=bad))
    codes = {i.code for i in t.validate() if i.severity == "error"}
    assert "street_order" in codes


def test_overlapping_hands_on_same_video_is_error():
    t = SyncTable("negreanu", FrameClock(30.0))
    t.add(make_entry("h1", 0, 500))
    t.add(make_entry("h2", 400, 900))  # starts before h1 ends
    errs = [i for i in t.validate() if i.code == "hand_frame_overlap"]
    assert errs and errs[0].severity == "error"


def test_overlap_allowed_across_different_videos():
    t = SyncTable("negreanu", FrameClock(30.0))
    t.add(make_entry("h1", 0, 500, video="ep1"))
    t.add(make_entry("h2", 0, 500, video="ep2"))
    assert [i for i in t.validate() if i.code == "hand_frame_overlap"] == []


def test_missing_streets_is_warning_not_error():
    t = SyncTable("negreanu", FrameClock(30.0))
    t.add(make_entry("h1", 0, 399))
    issues = t.validate()
    assert any(i.code == "no_street_boundaries" and i.severity == "warning"
               for i in issues)
    assert [i for i in issues if i.severity == "error"] == []


# --- drift detection -------------------------------------------------------

def test_drift_clean_sync_is_ok():
    # Video clock and HH clock agree exactly (slope 1, offset 100s).
    clock = FrameClock(30.0)
    t = SyncTable("negreanu", clock)
    for i in range(6):
        hh = 1000.0 + i * 60.0  # a hand every 60s of HH time
        video_seconds = hh - 100.0  # constant offset
        start_frame = clock.time_to_frame(video_seconds)
        t.add(make_entry(f"h{i}", start_frame, start_frame + 399, hh_time=hh))
    report = t.drift_report(tolerance_s=1.0)
    assert report.ok
    assert abs(report.slope - 1.0) < 1e-6
    assert report.max_abs_residual_s < 1.0


def test_drift_flags_single_mis_synced_hand():
    clock = FrameClock(30.0)
    t = SyncTable("negreanu", clock)
    for i in range(6):
        hh = 1000.0 + i * 60.0
        video_seconds = hh - 100.0
        if i == 3:
            video_seconds += 5.0  # this hand is mis-synced by 5 seconds
        start_frame = clock.time_to_frame(video_seconds)
        t.add(make_entry(f"h{i}", start_frame, start_frame + 399, hh_time=hh))
    report = t.drift_report(tolerance_s=1.0)
    assert not report.ok
    assert "h3" in report.flagged


def test_drift_requires_two_anchored_hands():
    t = SyncTable("negreanu", FrameClock(30.0))
    t.add(make_entry("h1", 0, 399, hh_time=1000.0))
    with pytest.raises(ValueError):
        t.drift_report()


# --- persistence + coverage ------------------------------------------------

def test_dataframe_round_trip():
    clock = FrameClock(30.0)
    t = SyncTable("negreanu", clock)
    t.add(make_entry("h1", 0, 399, hh_time=1000.0, streets=streets_in_order(0)))
    t.add(make_entry("h2", 500, 899, hh_time=1060.0, streets=streets_in_order(500)))

    df = t.to_dataframe()
    rebuilt = SyncTable.from_dataframe(df, clock)

    assert len(rebuilt) == 2
    assert rebuilt["h1"].start_frame == 0
    assert len(rebuilt["h1"].streets) == 4
    assert rebuilt["h2"].hh_start_time == 1060.0


def test_from_dataframe_rejects_multiple_players():
    clock = FrameClock(30.0)
    t = SyncTable("negreanu", clock)
    t.add(make_entry("h1", 0, 399))
    t.add(make_entry("h2", 500, 899))
    df = t.to_dataframe()
    # Corrupt one row so the frame spans two players.
    df.loc[df.index[-1], "player_id"] = "ivey"
    with pytest.raises(PermissionError):
        SyncTable.from_dataframe(df, clock)


def test_coverage_reports_gaps():
    clock = FrameClock(30.0)
    t = SyncTable("negreanu", clock)
    t.add(make_entry("h1", 0, 399, hh_time=1000.0))
    t.add(make_entry("orphan", 500, 899, hh_time=1060.0))

    hands = [
        HandHistory("h1", "negreanu", 1000.0,
                    [Action(Street.PREFLOP, "negreanu", ActionType.BET, 100,
                            wall_clock=1001.0)]),
        HandHistory("h2", "negreanu", 1120.0),  # not synced
    ]
    cov = t.coverage(hands)
    assert cov["synced"] == ["h1"]
    assert cov["missing_sync"] == ["h2"]
    assert cov["orphan_sync"] == ["orphan"]


def test_coverage_rejects_foreign_player():
    t = SyncTable("negreanu", FrameClock(30.0))
    foreign = [HandHistory("x", "ivey", 1.0)]
    with pytest.raises(PermissionError):
        t.coverage(foreign)
