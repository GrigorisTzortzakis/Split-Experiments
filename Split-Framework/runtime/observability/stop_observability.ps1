param(
    [string]$DockerPath = "docker"
)

$ErrorActionPreference = "Stop"
if ($null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$script:RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$hwinfoStateDir = Join-Path $PSScriptRoot "hwinfo_exporter"
$hwinfoRelayPidPath = Join-Path $hwinfoStateDir "relay.pid"
$promDapterPidPath = Join-Path $hwinfoStateDir "promdapter.pid"
$hwinfoPidPath = Join-Path $hwinfoStateDir "hwinfo.pid"

function Get-WslDockerPrefix {
    $wslCommand = Get-Command "wsl.exe" -ErrorAction SilentlyContinue
    if (-not $wslCommand) {
        return $null
    }

    $rawDistros = & $wslCommand.Source -l -q 2>$null
    $distros = @()
    foreach ($entry in $rawDistros) {
        $cleaned = ($entry -replace [char]0, "").Trim()
        if (-not [string]::IsNullOrWhiteSpace($cleaned)) {
            $distros += $cleaned
        }
    }

    $preferred = @("Ubuntu-24.04", "Ubuntu")
    $ordered = @($preferred | Where-Object { $distros -contains $_ }) + @($distros | Where-Object { $preferred -notcontains $_ })
    foreach ($distro in $ordered) {
        & $wslCommand.Source -d $distro -- sh -lc "command -v docker >/dev/null 2>&1"
        if ($LASTEXITCODE -eq 0) {
            return @($wslCommand.Source, "-d", $distro, "--", "docker")
        }
    }

    return $null
}

$dockerPrefix = $null
$dockerCommand = Get-Command $DockerPath -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $dockerPrefix = @($dockerCommand.Source)
} else {
    $dockerPrefix = Get-WslDockerPrefix
}

if ($null -eq $dockerPrefix) {
    throw "docker was not found on PATH, and no WSL distro with docker was detected."
}

$script:DockerPrefix = $dockerPrefix

function Invoke-DockerRaw {
    param([string[]]$DockerArgs)

    $prefixArgs = @()
    if ($script:DockerPrefix.Count -gt 1) {
        $prefixArgs = $script:DockerPrefix[1..($script:DockerPrefix.Count - 1)]
    }
    & $script:DockerPrefix[0] @prefixArgs @DockerArgs
}

function Get-DockerOutput {
    param([string[]]$DockerArgs)

    $result = Invoke-DockerRaw -DockerArgs $DockerArgs
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    return $result
}

function Remove-ContainerIfPresent {
    param([string]$Name)

    $existingName = Get-DockerOutput @("ps", "-a", "--filter", "name=^/${Name}$", "--format", "{{.Names}}")
    if ([string]::IsNullOrWhiteSpace($existingName) -or $existingName.Trim() -ne $Name) {
        return
    }

    Invoke-DockerRaw -DockerArgs @("rm", "-f", $Name) *> $null
}

function Stop-ManagedProcess {
    param([string]$PidPath)

    if (-not (Test-Path $PidPath)) {
        return
    }

    $pidRaw = Get-Content -Path $PidPath -ErrorAction SilentlyContinue | Select-Object -First 1
    $processId = 0
    if ([int]::TryParse([string]$pidRaw, [ref]$processId)) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
        } catch {
        }
    }

    Remove-Item -Force $PidPath -ErrorAction SilentlyContinue
}

$containerNames = @(
    "split-framework-observability-grafana",
    "split-framework-observability-prometheus",
    "split-framework-observability-kepler",
    "split-framework-observability-docker-stats",
    "split-framework-observability-pushgateway",
    "split-framework-observability-cadvisor",
    "split-framework-observability-scaphandre"
)

foreach ($name in $containerNames) {
    Remove-ContainerIfPresent -Name $name
}

foreach ($pidPath in @($hwinfoRelayPidPath, $promDapterPidPath, $hwinfoPidPath)) {
    Stop-ManagedProcess -PidPath $pidPath
}

foreach ($processName in @('HWiNFO64', 'PromDapterSvc')) {
    try {
        Get-Process -Name $processName -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    } catch {
    }
}

try {
    $service = Get-Service -Name "PromDapterSvc" -ErrorAction SilentlyContinue
    if ($service -and $service.Status -ne "Stopped") {
        Stop-Service -Name "PromDapterSvc" -Force -ErrorAction SilentlyContinue
    }
} catch {
}

Invoke-DockerRaw -DockerArgs @("network", "rm", "split-framework-observability") *> $null

Write-Output "Split-Framework observability containers were stopped."