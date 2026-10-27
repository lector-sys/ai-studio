"""A watchdog alert must stop only the explicitly owned inference process."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows GPU watchdog")
@pytest.mark.parametrize("manual_alert", [True, False])
def test_watchdog_stops_owned_pid_with_hidden_command_line(manual_alert):
    shell = shutil.which("powershell.exe")
    watch = Path(__file__).resolve().parents[1] / "tools" / "gpu_watch.ps1"
    children = [subprocess.Popen([shell, "-NoProfile", "-Command", "Start-Sleep 45"],
                                creationflags=subprocess.CREATE_NO_WINDOW) for _ in range(2)]
    try:
        script = f"""
function Get-CimInstance {{
    param($ClassName, $Filter)
    if ($ClassName -eq "Win32_VideoController") {{ [pscustomobject]@{{Name="Test Radeon"; Status="fault"}} }}
    if ($Filter -eq "Name='ffmpeg.exe'") {{
        [pscustomobject]@{{Name='ffmpeg.exe'; CommandLine='ffmpeg -c:v h264_amf'; ProcessId={children[1].pid}}}
    }}
}}
function Get-WinEvent {{}}
. '{watch}' -Minutes {1 if manual_alert else 0} -AlsoStopPid {children[0].pid}
"""
        result = subprocess.run([shell, "-NoProfile", "-Command", script],
                                capture_output=True, text=True, timeout=15,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode == (1 if manual_alert else 0), result.stderr
        try:
            children[0].wait(timeout=2)
        except subprocess.TimeoutExpired:
            pytest.fail("watchdog left the owned inference process running")
        assert children[1].poll() is None, "watchdog stopped an unrelated process"
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
