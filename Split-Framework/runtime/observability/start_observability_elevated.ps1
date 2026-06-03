param()

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
$startScript = Join-Path $scriptRoot "start_observability.ps1"

& $startScript -ProjectRoot $projectRoot
