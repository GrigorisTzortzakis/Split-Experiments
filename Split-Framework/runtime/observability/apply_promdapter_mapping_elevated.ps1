param(
    [Parameter(Mandatory = $true)]
    [string]$SourceMappingPath
)

$ErrorActionPreference = 'Stop'

$destinationDir = 'C:\ProgramData\PromDapter'
$destinationPath = Join-Path $destinationDir 'Prometheusmapping.yaml'

if (-not (Test-Path $SourceMappingPath)) {
    throw "Source mapping file not found: $SourceMappingPath"
}

New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
Copy-Item -Path $SourceMappingPath -Destination $destinationPath -Force
Restart-Service -Name 'PromDapterSvc' -Force -ErrorAction Stop

try {
    Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:10445/metrics/reset' -TimeoutSec 5 | Out-Null
} catch {
}

Write-Output "PromDapter mapping applied."
