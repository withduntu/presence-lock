"""
Presence state machine.

This module knows nothing about OpenCV, YOLO, or the camera - it only deals
with plain booleans/floats and time. That's deliberate: it keeps the hardest
logic in the project easy to unit test (see tests/test_presence.py) and easy
to reason about on its own.

States:
    PRESENT        -> normal, person is around
    ABSENT_PENDING -> window just flipped to "mostly absent", timer started
    WARNING        -> close to the lock delay, user has been warned
    LOCKED         -> lock delay exceeded, caller should lock the workstation
    COOLDOWN       -> pause after a lock (or camera-failure lock) before
                      monitoring resumes, so we don't immediately re-trigger
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum


class State(Enum):
    PRESENT = "PRESENT"
    ABSENT_PENDING = "ABSENT_PENDING"
    WARNING = "WARNING"
    LOCKED = "LOCKED"
    COOLDOWN = "COOLDOWN"


@dataclass
class PresenceEvent:
    """What the caller (main.py) should do after one update() call."""
    state: State
    action: str          # "none" | "warn" | "lock"
    message: str = ""     # human-readable reason, for logging


class PresenceTracker:
    def __init__(
        self,
        window_size: int,
        absent_ratio: float,
        lock_delay_seconds: float,
        warning_seconds_before_lock: float,
        cooldown_seconds: float,
        min_box_area_ratio: float,
        on_camera_failure: str,
        clock=time.monotonic,
    ):
        self.window_size = window_size
        self.absent_ratio = absent_ratio
        self.lock_delay_seconds = lock_delay_seconds
        self.warning_seconds_before_lock = warning_seconds_before_lock
        self.cooldown_seconds = cooldown_seconds
        self.min_box_area_ratio = min_box_area_ratio
        self.on_camera_failure = on_camera_failure
        self._clock = clock  # injected so tests can control time without real sleeps

        self._window = deque(maxlen=window_size)
        self.state = State.PRESENT
        self._absence_started_at = None
        self._cooldown_started_at = None

    def update(self, detected: bool, box_area_ratio: float = 1.0) -> PresenceEvent:
        """
        Feed one sampled frame's result in. `detected` is whether YOLO saw a
        person at all; `box_area_ratio` is that person's box area divided by
        the frame area. A detection only counts as "present" if the box is
        big enough (filters out people passing by in the background).
        """
        now = self._clock()
        effective_present = detected and box_area_ratio >= self.min_box_area_ratio
        self._window.append(effective_present)

        if self.state == State.COOLDOWN:
            return self._handle_cooldown(now)

        window_absent = self._window_is_absent()

        if self.state == State.PRESENT:
            return self._handle_present(window_absent, now)

        if self.state in (State.ABSENT_PENDING, State.WARNING):
            return self._handle_absent_states(window_absent, now)

        # LOCKED: we hold here until main.py calls enter_cooldown() right
        # after acting on the "lock" event.
        return PresenceEvent(self.state, "none")

    def handle_camera_failure(self) -> PresenceEvent:
        """
        Called by main.py when frames can't be read for too long. Doesn't
        touch the sliding window since we have no visual data to add.
        """
        if self.on_camera_failure == "lock":
            self.state = State.LOCKED
            return PresenceEvent(self.state, "lock", "camera failure - locking per config")
        return PresenceEvent(self.state, "none", "camera failure - ignoring per config")

    def enter_cooldown(self) -> None:
        """Call this right after acting on a 'lock' event."""
        self.state = State.COOLDOWN
        self._cooldown_started_at = self._clock()
        self._absence_started_at = None

    def _window_is_absent(self) -> bool:
        if len(self._window) < self.window_size:
            return False  # not enough samples yet - give the benefit of the doubt
        absent_count = sum(1 for present in self._window if not present)
        return (absent_count / len(self._window)) >= self.absent_ratio

    def _handle_cooldown(self, now: float) -> PresenceEvent:
        if now - self._cooldown_started_at >= self.cooldown_seconds:
            self.state = State.PRESENT
            self._window.clear()
            return PresenceEvent(self.state, "none", "cooldown ended, monitoring resumed")
        return PresenceEvent(self.state, "none")

    def _handle_present(self, window_absent: bool, now: float) -> PresenceEvent:
        if window_absent:
            self.state = State.ABSENT_PENDING
            self._absence_started_at = now
            return PresenceEvent(self.state, "none", "absence detected, timer started")
        return PresenceEvent(self.state, "none")

    def _handle_absent_states(self, window_absent: bool, now: float) -> PresenceEvent:
        if not window_absent:
            self.state = State.PRESENT
            self._absence_started_at = None
            return PresenceEvent(self.state, "none", "presence returned, timer cancelled")

        elapsed = now - self._absence_started_at
        warn_at = self.lock_delay_seconds - self.warning_seconds_before_lock

        if elapsed >= self.lock_delay_seconds:
            self.state = State.LOCKED
            return PresenceEvent(self.state, "lock", "absence exceeded lock delay")

        if self.state == State.ABSENT_PENDING and elapsed >= warn_at:
            self.state = State.WARNING
            return PresenceEvent(self.state, "warn", "warning before lock")

        return PresenceEvent(self.state, "none")
