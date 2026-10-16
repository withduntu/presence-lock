"""
Entry point. Loads config, opens the camera, loads YOLO once, and runs the
main sampling loop: read a frame, detect, feed presence.py, act on warnings
and lock events. Press Ctrl+C anytime (or 'q' if show_preview is on) to stop.

Run from the project root with:
    python -m src.main
"""

import time
from pathlib import Path

import cv2
import yaml

from .camera import Camera
from .detector import Detector
from .locker import lock_workstation
from .logger import setup_logger
from .notifier import notify
from .presence import PresenceTracker

BASE_DIR = Path(__file__).resolve().parent.parent  # project root, one level above src/


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def act_on_event(event, tracker, config, logger) -> None:
    """Logs the event and performs the action it carries, if any."""
    if event.message:
        logger.info("[%s] %s", event.state.value, event.message)

    if event.action == "warn":
        notify("You seem to have stepped away - locking soon.")
        logger.info("Warning shown to user")

    elif event.action == "lock":
        if config["dry_run"]:
            logger.info("DRY RUN: WOULD LOCK workstation now")
        else:
            lock_workstation()
            logger.info("Workstation locked")
        tracker.enter_cooldown()


def main() -> None:
    config = load_config()
    logger = setup_logger(str(BASE_DIR / config["log_file"]))
    logger.info("presence-lock started (dry_run=%s)", config["dry_run"])

    camera = Camera(config["camera_index"])
    if not camera.open():
        logger.error("Could not open camera index %s", config["camera_index"])
        return

    detector = Detector(
        model_path=config["model_path"],
        imgsz=config["imgsz"],
        confidence=config["confidence"],
    )

    tracker = PresenceTracker(
        window_size=config["window_size"],
        absent_ratio=config["absent_ratio"],
        lock_delay_seconds=config["lock_delay_seconds"],
        warning_seconds_before_lock=config["warning_seconds_before_lock"],
        cooldown_seconds=config["cooldown_seconds"],
        min_box_area_ratio=config["min_box_area_ratio"],
        on_camera_failure=config["on_camera_failure"],
    )

    sample_interval = 1.0 / config["sample_fps"]
    consecutive_failures = 0
    # Treat ~3 seconds' worth of back-to-back failed reads as a real camera
    # failure, rather than reacting to a single dropped frame.
    failure_threshold = max(1, int(config["sample_fps"] * 3))

    try:
        while True:
            loop_start = time.monotonic()
            frame = camera.read_frame()

            if frame is None:
                consecutive_failures += 1
                if consecutive_failures == failure_threshold:
                    logger.warning(
                        "Camera failure: %s consecutive failed reads",
                        consecutive_failures,
                    )
                    event = tracker.handle_camera_failure()
                    act_on_event(event, tracker, config, logger)
                time.sleep(sample_interval)
                continue

            consecutive_failures = 0
            detection = detector.detect(frame)
            event = tracker.update(detection.present, detection.box_area_ratio)
            act_on_event(event, tracker, config, logger)

            if config["show_preview"]:
                label = (
                    f"{tracker.state.value} "
                    f"present={detection.present} area={detection.box_area_ratio:.3f}"
                )
                cv2.putText(
                    frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.imshow("presence-lock preview (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, sample_interval - elapsed))

    except KeyboardInterrupt:
        logger.info("Shutdown requested (Ctrl+C)")

    finally:
        camera.release()
        if config["show_preview"]:
            cv2.destroyAllWindows()
        logger.info("presence-lock stopped")


if __name__ == "__main__":
    main()
