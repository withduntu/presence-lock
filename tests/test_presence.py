"""
Unit tests for the presence state machine, using plain True/False sequences
and a fake clock - no camera, no YOLO, no real sleeping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.presence import PresenceTracker, State


class FakeClock:
    """A controllable stand-in for time.monotonic()."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_tracker(clock, **overrides):
    defaults = dict(
        window_size=5,
        absent_ratio=0.8,      # 4 of 5 recent frames absent => window is ABSENT
        lock_delay_seconds=8,
        warning_seconds_before_lock=3,
        cooldown_seconds=10,
        min_box_area_ratio=0.08,
        on_camera_failure="lock",
        clock=clock,
    )
    defaults.update(overrides)
    return PresenceTracker(**defaults)


def test_brief_look_away_does_not_lock():
    clock = FakeClock()
    tracker = make_tracker(clock)

    # One missed frame in a window of 5 never reaches the 80% absent ratio.
    sequence = [True, True, False, True, True, True, False, True]
    for present in sequence:
        event = tracker.update(present)
        assert event.action == "none"
        assert tracker.state == State.PRESENT


def test_sustained_absence_locks():
    clock = FakeClock()
    tracker = make_tracker(clock)

    # Fill the window with absence so the state moves to ABSENT_PENDING.
    for _ in range(5):
        event = tracker.update(False)
    assert tracker.state == State.ABSENT_PENDING

    # Before the warning threshold (lock_delay - warning = 5s), nothing happens.
    clock.advance(4)
    event = tracker.update(False)
    assert event.action == "none"
    assert tracker.state == State.ABSENT_PENDING

    # Crossing into the warning window triggers the warning exactly once.
    clock.advance(1)  # now at 5s elapsed
    event = tracker.update(False)
    assert event.action == "warn"
    assert tracker.state == State.WARNING

    # Crossing the full lock delay (8s) triggers the lock.
    clock.advance(3)  # now at 8s elapsed
    event = tracker.update(False)
    assert event.action == "lock"
    assert tracker.state == State.LOCKED


def test_return_during_warning_cancels_lock():
    clock = FakeClock()
    tracker = make_tracker(clock)

    for _ in range(5):
        tracker.update(False)
    assert tracker.state == State.ABSENT_PENDING

    clock.advance(5)
    event = tracker.update(False)
    assert event.action == "warn"
    assert tracker.state == State.WARNING

    # User sits back down: enough "present" frames to flip the window back.
    for _ in range(5):
        event = tracker.update(True)

    assert event.action == "none"
    assert tracker.state == State.PRESENT

    # And it should NOT lock even if we advance well past the lock delay.
    clock.advance(10)
    event = tracker.update(True)
    assert event.action == "none"
    assert tracker.state == State.PRESENT


def test_cooldown_prevents_immediate_relock():
    clock = FakeClock()
    tracker = make_tracker(clock)

    for _ in range(5):
        tracker.update(False)
    clock.advance(8)
    event = tracker.update(False)
    assert event.action == "lock"

    tracker.enter_cooldown()
    assert tracker.state == State.COOLDOWN

    # Still absent, but cooldown should absorb it - no re-lock.
    for _ in range(5):
        event = tracker.update(False)
        assert event.action == "none"
        assert tracker.state == State.COOLDOWN

    # Cooldown not elapsed yet (default 10s in these tests).
    clock.advance(9)
    event = tracker.update(False)
    assert tracker.state == State.COOLDOWN

    # Now cooldown elapses - back to normal monitoring.
    clock.advance(2)
    event = tracker.update(False)
    assert tracker.state == State.PRESENT
    assert event.action == "none"


def test_camera_failure_locks_when_configured():
    clock = FakeClock()
    tracker = make_tracker(clock, on_camera_failure="lock")

    event = tracker.handle_camera_failure()
    assert event.action == "lock"
    assert tracker.state == State.LOCKED


def test_camera_failure_ignored_when_configured():
    clock = FakeClock()
    tracker = make_tracker(clock, on_camera_failure="ignore")

    event = tracker.handle_camera_failure()
    assert event.action == "none"
    assert tracker.state == State.PRESENT  # unchanged


def test_small_box_ignored():
    clock = FakeClock()
    tracker = make_tracker(clock, min_box_area_ratio=0.08)

    # Person "detected" every frame, but the box is tiny (e.g. someone far
    # in the background) - should count as absent, not present.
    for _ in range(5):
        event = tracker.update(True, box_area_ratio=0.01)
    assert tracker.state == State.ABSENT_PENDING

    # Reset and confirm a big-enough box is correctly treated as present.
    tracker = make_tracker(clock, min_box_area_ratio=0.08)
    for _ in range(5):
        event = tracker.update(True, box_area_ratio=0.5)
    assert tracker.state == State.PRESENT
