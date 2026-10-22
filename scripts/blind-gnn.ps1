param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('register','export','preview','submit','status','recover','confirm')]
    [string]$Action,
    [ValidateSet('chatgpt','claude')][string]$Provider,
    [ValidateSet('independent','rsi')][string]$Arm,
    [string]$Model,
    [string]$Submission,
    [string]$Campaign = 'runs/blind-g1-campaign-01'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$arguments = @('-d','Ubuntu','--','python3','-m','rsi.blind_campaign',$Action,'--campaign',$Campaign)
if ($Provider) { $arguments += @('--provider',$Provider) }
if ($Arm) { $arguments += @('--arm',$Arm) }
if ($Model) { $arguments += @('--model',$Model) }
if ($Submission) {
    $resolvedSubmission = (Resolve-Path -LiteralPath $Submission).Path
    $linuxSubmission = (& wsl -d Ubuntu -- wslpath -a $resolvedSubmission).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not translate the submission path' }
    $arguments += @('--submission',$linuxSubmission)
}
Push-Location -LiteralPath $projectRoot
try {
    & wsl @arguments
    if ($LASTEXITCODE -ne 0) { throw "Controller exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
