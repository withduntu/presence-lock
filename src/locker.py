"""
The only module that touches locking the workstation. It never touches
authentication - unlocking always happens through the normal Windows
password. On non-Windows platforms this is a no-op stub.
"""

import ctypes
import platform


def lock_workstation() -> bool:
    """Locks the current session. Returns True if a lock call was made."""
    system = platform.system()

    if system == "Windows":
        ctypes.windll.user32.LockWorkStation()
        return True

    # macOS / Linux: not implemented. This project targets Windows laptops.
    return False
