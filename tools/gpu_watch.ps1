# GPU health watchdog for studio renders on the AMD card.
# Prints a status line every $Every seconds and an ALERT line (then kills AMF ffmpeg encodes) if Windows logs a
# display-driver reset (TDR 4101/4116/4117), the adapter status leaves "OK", or dedicated VRAM passes $MaxVramMB.
# -AlsoStopPid <owned-PID> stops only the recorded process; -AlsoStop "main.py --listen" also stops GPU inference processes (e.g. a task-owned ComfyUI) whose command line matches.
param([int]$Every = 10, [int]$MaxVramMB = 14000, [double]$Minutes = 240, [string]$AlsoStop = "", [int[]]$AlsoStopPid = @())

# Explicit PIDs cover Store Python processes whose command lines Windows hides.
$ownedInference = @($AlsoStopPid | ForEach-Object { Get-Process -Id $_ -ErrorAction Stop } | Select-Object Id, StartTime)
$start = Get-Date
$deadline = $start.AddMinutes($Minutes)  # bounded session: never watches (or runs) forever
$amd = (Get-CimInstance Win32_VideoController | Where-Object Name -like "*Radeon*" | Select-Object -First 1).Name

function Stop-GpuEncode($why) {
    Write-Output "ALERT $(Get-Date -Format HH:mm:ss) $why -> stopping AMF encodes"
    if (!$ownedInference.Count) {  # Keep legacy AMF monitoring; explicit ownership never stops other jobs.
        Get-CimInstance Win32_Process -Filter "Name='ffmpeg.exe'" |
            Where-Object { $_.CommandLine -match "_amf" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
    foreach ($owned in $ownedInference) {
        $current = Get-Process -Id $owned.Id -ErrorAction SilentlyContinue
        if ($current -and $current.StartTime -eq $owned.StartTime) {
            Write-Output "ALERT stopping owned pid $($owned.Id)"
            Stop-Process -Id $owned.Id -Force -ErrorAction SilentlyContinue
        }
    }
    if ($AlsoStop) {
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -and $_.CommandLine.Contains($AlsoStop) } |
            ForEach-Object { Write-Output "ALERT stopping pid $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}

while ((Get-Date) -lt $deadline) {
    $tdr = Get-WinEvent -FilterHashtable @{LogName = 'System'; Id = 4101, 4116, 4117; StartTime = $start} -ErrorAction SilentlyContinue
    if ($tdr) { Stop-GpuEncode "display driver reset (event $($tdr[0].Id))"; exit 1 }
    $status = (Get-CimInstance Win32_VideoController | Where-Object Name -eq $amd).Status
    if ($status -ne "OK") { Stop-GpuEncode "adapter status '$status'"; exit 1 }

    $s = (Get-Counter '\GPU Engine(*engtype_VideoEncode)\Utilization Percentage', '\GPU Engine(*engtype_3D)\Utilization Percentage',
          '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
    $enc = ($s | Where-Object Path -like "*videoencode*" | Measure-Object CookedValue -Maximum).Maximum
    $g3d = ($s | Where-Object Path -like "*engtype_3d*" | Measure-Object CookedValue -Maximum).Maximum
    $vram = [math]::Round((($s | Where-Object Path -like "*dedicated usage*" | Measure-Object CookedValue -Maximum).Maximum) / 1MB)
    if ($vram -gt $MaxVramMB) { Stop-GpuEncode "VRAM $vram MB > $MaxVramMB MB"; exit 1 }
    $ff = @(Get-CimInstance Win32_Process -Filter "Name='ffmpeg.exe'" | Where-Object { $_.CommandLine -match "_amf" }).Count
    Write-Output ("{0} gpu ok | encode {1:N0}% | 3d {2:N0}% | vram {3} MB | amf encodes {4}" -f (Get-Date -Format HH:mm:ss), $enc, $g3d, $vram, $ff)
    Start-Sleep -Seconds $Every
}
if ($ownedInference.Count) { Stop-GpuEncode "session time limit" }
Write-Output "watch ended (time limit)"
