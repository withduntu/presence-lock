"""
Shows a short warning before the workstation locks. Tries a Windows toast
notification; if that fails for any reason (package missing, non-Windows,
anything else) it falls back to printing to the console so a notification
problem never crashes the app.
"""

import platform


def notify(message: str) -> None:
    if platform.system() == "Windows":
        try:
            from win11toast import notify as toast_notify
            toast_notify(title="presence-lock", body=message)
            return
        except Exception:
            pass  # fall through to console

    print(f"[NOTICE] {message}")
