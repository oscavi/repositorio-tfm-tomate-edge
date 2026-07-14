$RunRoot = "F:\TFM experimento\training\runs\pareto_sweep_2026_05_21"

Write-Host "=== Python processes ==="
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*pareto*" -or $_.CommandLine -like "*train_cls_hparams*" } |
    Select-Object ProcessId, ParentProcessId, CommandLine |
    Format-List

Write-Host "=== GPU ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

Write-Host "=== Status ==="
$Status = Join-Path $RunRoot "status.json"
if (Test-Path $Status) {
    Get-Content -LiteralPath $Status
} else {
    Write-Host "No status.json found yet."
}

Write-Host "=== Latest log tail ==="
$LatestLog = Get-ChildItem -LiteralPath (Join-Path $RunRoot "logs") -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($LatestLog) {
    Write-Host $LatestLog.FullName
    Get-Content -LiteralPath $LatestLog.FullName -Tail 60
} else {
    Write-Host "No logs found yet."
}
