"""Check that process gating recognizes live processes and fails closed."""

import os
import sys

import pytest

from scripts.check_micro_mlip_resource_gate import process_state


def test_current_launcher_cannot_be_treated_as_absent():
    assert process_state(os.getpid()) == "ACTIVE"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process inspection")
def test_unreadable_windows_process_does_not_authorize_admission(monkeypatch):
    import psutil

    def unreadable(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", unreadable)
    with pytest.raises(psutil.AccessDenied):
        process_state(os.getpid())
