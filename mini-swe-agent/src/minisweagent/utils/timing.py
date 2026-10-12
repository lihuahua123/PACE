import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any


def elapsed_since(start_time: float) -> float:
    return time.perf_counter() - start_time


def fail_on_threshold_exceeded(
    *,
    operation: str,
    elapsed_sec: float,
    threshold_sec: float | int | None,
    logger: logging.Logger | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if not threshold_sec or threshold_sec <= 0 or elapsed_sec <= threshold_sec:
        return

    report_threshold_exceeded(
        operation=operation,
        elapsed_sec=elapsed_sec,
        threshold_sec=threshold_sec,
        logger=logger,
        details=details,
    )


def report_threshold_exceeded(
    *,
    operation: str,
    elapsed_sec: float,
    threshold_sec: float | int | None,
    logger: logging.Logger | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if not threshold_sec or threshold_sec <= 0:
        return

    details_text = ""
    if details:
        details_text = " " + " ".join(f"{key}={value!r}" for key, value in details.items())
    message = (
        f"WARNING: {operation} exceeded time threshold: "
        f"elapsed={elapsed_sec:.2f}s threshold={float(threshold_sec):.2f}s.{details_text}"
    )
    if logger is not None:
        logger.warning(message)
    print(message, file=sys.stderr, flush=True)


def _watchdog_interval(threshold_sec: float | int) -> float:
    configured = os.getenv("MSWEA_STUCK_REPORT_INTERVAL", "")
    try:
        interval = float(configured) if configured else float(threshold_sec)
    except ValueError:
        interval = float(threshold_sec)
    return max(1.0, interval)


@contextmanager
def threshold_watchdog(
    *,
    operation: str,
    start_time: float,
    threshold_sec: float | int | None,
    logger: logging.Logger | None = None,
    details: dict[str, Any] | None = None,
):
    if not threshold_sec or threshold_sec <= 0:
        yield
        return

    stop_event = threading.Event()

    def report_while_waiting() -> None:
        threshold = float(threshold_sec)
        interval = _watchdog_interval(threshold)
        if stop_event.wait(threshold):
            return
        while not stop_event.is_set():
            report_threshold_exceeded(
                operation=operation,
                elapsed_sec=elapsed_since(start_time),
                threshold_sec=threshold,
                logger=logger,
                details=details,
            )
            if stop_event.wait(interval):
                return

    thread = threading.Thread(target=report_while_waiting, name=f"{operation}-watchdog", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
