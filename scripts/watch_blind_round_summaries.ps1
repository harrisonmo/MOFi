param(
    [string]$RunRoot = '',
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $RunRoot) { $RunRoot = Join-Path $projectRoot 'runs/blind-codex-sol-high' }
$RunRoot = [IO.Path]::GetFullPath($RunRoot)
$summaryRoot = Join-Path $RunRoot 'summaries'
New-Item -ItemType Directory -Path $summaryRoot -Force | Out-Null
$seen = @{}

function Write-RoundSummaries {
    $updated = $false
    foreach ($directory in (Get-ChildItem -LiteralPath $RunRoot -Directory)) {
        if ($directory.Name -notmatch '^(independent|rsi)-[1-3]$') { continue }
        $recordPath = Join-Path $directory.FullName 'call.json'
        if (-not (Test-Path -LiteralPath $recordPath)) { continue }
        try {
            $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
            if (-not $record.feedback) { continue }
            $digest = (Get-FileHash -LiteralPath $recordPath -Algorithm SHA256).Hash
            if ($seen[$directory.Name] -eq $digest) { continue }
            $feedback = $record.feedback
            $mae = if ($feedback.metrics) { '{0:F6}' -f [double]$feedback.metrics.mean_mae } else { 'unavailable' }
            $accepted = if ($null -ne $feedback.accepted) { [string]$feedback.accepted } else { 'unavailable' }
            $reason = if ($feedback.reason) { [string]$feedback.reason } else { 'none' }
            $seconds = '{0:F1}' -f [double]$record.seconds
            $text = @(
                "## $($directory.Name)", '',
                "- Status: $($feedback.status)",
                "- Development MAE: $mae mmol/g",
                "- Accepted: $accepted",
                "- Failure reason: $reason",
                "- Proposal time: $seconds seconds",
                "- Model: $($record.requested_model); reasoning: $($record.reasoning_effort)",
                "- Tool events: $(@($record.unexpected_item_types).Count)",
                "- Input/output tokens: $($record.usage.input_tokens) / $($record.usage.output_tokens)", ''
            ) -join "`n"
            [IO.File]::WriteAllText((Join-Path $summaryRoot ($directory.Name + '.md')), $text)
            $seen[$directory.Name] = $digest
            $updated = $true
            Write-Output $text
        } catch {
            # A writer may still be replacing its JSON file; the next filesystem event retries it.
            continue
        }
    }
    if ($updated) {
        $parts = @('# Blind-G1 round summaries', '', 'Generated on filesystem events; no model calls or scientific reruns.', '')
        foreach ($file in (Get-ChildItem -LiteralPath $summaryRoot -Filter '*.md' | Sort-Object Name)) {
            $parts += Get-Content -LiteralPath $file.FullName -Raw
        }
        [IO.File]::WriteAllText((Join-Path $RunRoot 'SUMMARY.md'), ($parts -join "`n"))
    }
}

if ($Once) { Write-RoundSummaries; exit }
$watcher = [IO.FileSystemWatcher]::new($RunRoot, '*.json')
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [IO.NotifyFilters]::LastWrite -bor [IO.NotifyFilters]::FileName -bor [IO.NotifyFilters]::Size
$ids = @('mofi-round-changed', 'mofi-round-created', 'mofi-round-renamed')
Register-ObjectEvent $watcher Changed -SourceIdentifier $ids[0] | Out-Null
Register-ObjectEvent $watcher Created -SourceIdentifier $ids[1] | Out-Null
Register-ObjectEvent $watcher Renamed -SourceIdentifier $ids[2] | Out-Null
$watcher.EnableRaisingEvents = $true
try {
    Write-RoundSummaries
    while ($true) {
        $event = Wait-Event -Timeout 60
        if ($event) {
            Remove-Event -EventIdentifier $event.EventIdentifier
            Write-RoundSummaries
        }
        if (Test-Path -LiteralPath (Join-Path $RunRoot 'confirmation.json')) {
            try {
                $final = Get-Content -LiteralPath (Join-Path $RunRoot 'confirmation.json') -Raw | ConvertFrom-Json
            } catch { continue }
            [IO.File]::WriteAllText((Join-Path $RunRoot 'CONFIRMATION_SUMMARY.md'),
                "# Confirmation summary`n`nStatus: $($final.status)`n`n$($final.aggregate | ConvertTo-Json)`n")
            Push-Location -LiteralPath $projectRoot
            try {
                & (Join-Path $projectRoot 'Crystal-Builder/.venv/Scripts/python.exe') -m scripts.report_blind_codex
                if ($LASTEXITCODE -eq 0) {
                    $env:GIT_CONFIG_COUNT = '1'
                    $env:GIT_CONFIG_KEY_0 = 'safe.directory'
                    $env:GIT_CONFIG_VALUE_0 = $projectRoot.Replace('\','/')
                    $branch = & git rev-parse --abbrev-ref HEAD
                    if ($branch -eq 'main') {
                        & git diff --check -- docs/reports/BLIND_G1_SOL_RESULT.json docs/reports/BLIND_G1_SOL_RESULT.md
                        if ($LASTEXITCODE -eq 0) {
                            & git add -- docs/reports/BLIND_G1_SOL_RESULT.json docs/reports/BLIND_G1_SOL_RESULT.md
                            & git commit --only -m 'Record GPT-6.1 Sol blind pilot results' -- docs/reports/BLIND_G1_SOL_RESULT.json docs/reports/BLIND_G1_SOL_RESULT.md
                        }
                    }
                }
            } finally { Pop-Location }
            break
        }
    }
} finally {
    foreach ($id in $ids) { Unregister-Event -SourceIdentifier $id -ErrorAction SilentlyContinue }
    $watcher.Dispose()
}
