param([Parameter(Mandatory=$true)][string]$RequestPath)
$ErrorActionPreference = 'Stop'
$request = Get-Content -LiteralPath $RequestPath -Raw | ConvertFrom-Json
$root = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($RequestPath))
$recordPath = Join-Path $root ($request.round + '/call.json')
$watcher = [IO.FileSystemWatcher]::new($root, 'call.json')
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [IO.NotifyFilters]::LastWrite -bor [IO.NotifyFilters]::Size
$eventIds = @('mofi-pause-changed', 'mofi-pause-created')
Register-ObjectEvent $watcher Changed -SourceIdentifier $eventIds[0] | Out-Null
Register-ObjectEvent $watcher Created -SourceIdentifier $eventIds[1] | Out-Null
$watcher.EnableRaisingEvents = $true
try {
    while ($true) {
        $finished = $false
        try {
            $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
            $finished = $null -ne $record.feedback
        } catch { }
        if ($finished) {
            foreach ($owned in $request.owned_roots) {
                $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($owned.pid)"
                if ($current -and $current.CommandLine -like '*-m scripts.run_blind_codex_cli --run*' -and
                    $current.CreationDate.ToUniversalTime().Ticks -eq ([DateTime]$owned.created_utc).ToUniversalTime().Ticks) {
                    & taskkill /PID $owned.pid /T /F | Out-Null
                    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the owned experiment driver' }
                }
            }
            $result = [ordered]@{status='paused';round=$request.round;feedback=$record.feedback;
                paused_at_utc=[DateTime]::UtcNow.ToString('o');reason='User requested pause after the current round'}
            $result | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 (Join-Path $root 'PAUSED.json')
            Write-Output ($result | ConvertTo-Json -Depth 20)
            break
        }
        # Native event waiting does not poll the scientific controller or call a model.
        $event = Wait-Event -Timeout 60
        if ($event) { Remove-Event -EventIdentifier $event.EventIdentifier }
    }
} finally {
    foreach ($id in $eventIds) { Unregister-Event -SourceIdentifier $id -ErrorAction SilentlyContinue }
    $watcher.Dispose()
}
