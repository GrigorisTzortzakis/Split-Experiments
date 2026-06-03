param()

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
$stopScript = Join-Path $scriptRoot "stop_observability.ps1"
$startScript = Join-Path $scriptRoot "start_observability.ps1"

& $stopScript
& $startScript -ProjectRoot $projectRoot