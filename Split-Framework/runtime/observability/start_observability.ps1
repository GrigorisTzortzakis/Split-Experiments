param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [string]$DockerPath = "docker"
)

$ErrorActionPreference = "Stop"
if ($null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$networkName = "split-framework-observability"
$prometheusName = "split-framework-observability-prometheus"
$grafanaName = "split-framework-observability-grafana"
$keplerName = "split-framework-observability-kepler"
$dockerStatsName = "split-framework-observability-docker-stats"
$pushgatewayName = "split-framework-observability-pushgateway"
$grafanaVolume = "split-framework-observability-grafana-data"
$dockerStatsImage = "split-framework-docker-stats-exporter:latest"
$promDapterDownloadUrl = "https://github.com/kallex/PromDapter/releases/download/v2022.0830.30-beta/PromDapter_2022.0830.30.zip"

$prometheusConfig = Join-Path $ProjectRoot "runtime/observability/prometheus.yml"
$keplerConfig = Join-Path $ProjectRoot "runtime/observability/kepler/config.yaml"
$dockerStatsContext = Join-Path $ProjectRoot "runtime/observability/docker_stats_exporter"
$hwinfoRelayScript = Join-Path $ProjectRoot "runtime/observability/hwinfo_exporter/exporter.ps1"
$hwinfoMappingConfig = Join-Path $ProjectRoot "runtime/observability/hwinfo_exporter/Prometheusmapping.yaml"
$hwinfoStateDir = Join-Path $ProjectRoot "runtime/observability/hwinfo_exporter"
$hwinfoVendorDir = Join-Path $hwinfoStateDir "vendor"
$promDapterDir = Join-Path $hwinfoVendorDir "PromDapter"
$promDapterZip = Join-Path $hwinfoStateDir "PromDapter.zip"
$promDapterInstalledDir = "C:\Program Files\PromDapter"
$hwinfoRelayPidPath = Join-Path $hwinfoStateDir "relay.pid"
$hwinfoRelayStdOutPath = Join-Path $hwinfoStateDir "relay.stdout.log"
$hwinfoRelayStdErrPath = Join-Path $hwinfoStateDir "relay.stderr.log"
$promDapterPidPath = Join-Path $hwinfoStateDir "promdapter.pid"
$promDapterStdOutPath = Join-Path $hwinfoStateDir "promdapter.stdout.log"
$promDapterStdErrPath = Join-Path $hwinfoStateDir "promdapter.stderr.log"
$hwinfoPidPath = Join-Path $hwinfoStateDir "hwinfo.pid"
$grafanaDatasource = Join-Path $ProjectRoot "runtime/observability/grafana/datasources.yml"
$grafanaDashboards = Join-Path $ProjectRoot "runtime/observability/grafana/dashboards.yml"
$grafanaDashboard = Join-Path $ProjectRoot "runtime/observability/grafana/split-framework-observability-live.json"

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

function Convert-HostPathForDocker {
    param([string]$HostPath)

    if (-not $script:UseWslDocker) {
        return $HostPath
    }

    $normalized = $HostPath -replace "\\", "/"
    if ($normalized -match "^([A-Za-z]):/(.*)$") {
        return "/mnt/$($matches[1].ToLower())/$($matches[2])"
    }

    return $normalized
}

$dockerPrefix = $null
$dockerCommand = Get-Command $DockerPath -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $dockerPrefix = @($dockerCommand.Source)
    $script:UseWslDocker = $false
} else {
    $dockerPrefix = Get-WslDockerPrefix
    $script:UseWslDocker = ($null -ne $dockerPrefix)
}

if ($null -eq $dockerPrefix) {
    throw "docker was not found on PATH, and no WSL distro with docker was detected."
}

$script:DockerPrefix = $dockerPrefix
$prometheusMount = Convert-HostPathForDocker -HostPath $prometheusConfig
$keplerMount = Convert-HostPathForDocker -HostPath $keplerConfig
$dockerStatsContextMount = Convert-HostPathForDocker -HostPath $dockerStatsContext
$grafanaDatasourceMount = Convert-HostPathForDocker -HostPath $grafanaDatasource
$grafanaDashboardsMount = Convert-HostPathForDocker -HostPath $grafanaDashboards
$grafanaDashboardMount = Convert-HostPathForDocker -HostPath $grafanaDashboard

if ($script:UseWslDocker) {
    Write-Warning "WSL does not expose real RAPL/hwmon CPU power sensors in this setup. Kepler will use its fake CPU fallback so GPU and container attribution stay available, but CPU energy values are estimates inside WSL."
}

foreach ($requiredPath in @($prometheusConfig, $keplerConfig, $dockerStatsContext, $grafanaDatasource, $grafanaDashboards, $grafanaDashboard)) {
    if (-not (Test-Path $requiredPath)) {
        throw "Missing observability asset: $requiredPath"
    }
}

foreach ($requiredPath in @($hwinfoRelayScript, $hwinfoMappingConfig)) {
    if (-not (Test-Path $requiredPath)) {
        throw "Missing observability asset: $requiredPath"
    }
}

function Invoke-DockerRaw {
    param([string[]]$DockerArgs)

    $prefixArgs = @()
    if ($script:DockerPrefix.Count -gt 1) {
        $prefixArgs = $script:DockerPrefix[1..($script:DockerPrefix.Count - 1)]
    }
    & $script:DockerPrefix[0] @prefixArgs @DockerArgs
}

function Invoke-Docker {
    param([string[]]$DockerArgs)

    Invoke-DockerRaw -DockerArgs $DockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($DockerArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-DockerOutput {
    param([string[]]$DockerArgs)

    $result = Invoke-DockerRaw -DockerArgs $DockerArgs
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    return $result
}

function Test-ContainerRunning {
    param([string]$Name)

    $runningName = Get-DockerOutput @("ps", "--filter", "name=^/${Name}$", "--format", "{{.Names}}")
    if ([string]::IsNullOrWhiteSpace($runningName)) {
        return $false
    }

    return ($runningName.Trim() -eq $Name)
}

function Remove-ContainerIfPresent {
    param([string]$Name)

    $existingName = Get-DockerOutput @("ps", "-a", "--filter", "name=^/${Name}$", "--format", "{{.Names}}")
    if ([string]::IsNullOrWhiteSpace($existingName) -or $existingName.Trim() -ne $Name) {
        return
    }

    Invoke-DockerRaw -DockerArgs @("rm", "-f", $Name) *> $null
}

function Test-ProcessAlive {
    param([int]$ProcessId)

    try {
        $null = Get-Process -Id $ProcessId -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Test-HttpEndpoint {
    param([string]$Uri)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 5
        return -not [string]::IsNullOrWhiteSpace($response.Content)
    } catch {
        return $false
    }
}

function Get-HwinfoExecutable {
    $candidates = @(
        "C:\Program Files\HWiNFO64\HWiNFO64.EXE",
        "C:\Program Files\HWiNFO\HWiNFO64.EXE",
        "C:\Program Files (x86)\HWiNFO64\HWiNFO64.EXE"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Get-WingetExecutable {
    $candidates = @(
        "C:\Users\$env:USERNAME\AppData\Local\Microsoft\WindowsApps\winget.exe",
        "C:\Program Files\WindowsApps\Microsoft.DesktopAppInstaller_8wekyb3d8bbwe\winget.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $command = Get-Command "winget.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $command = Get-Command "winget" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    return $null
}

function Install-HwinfoWithWinget {
    $wingetExecutable = Get-WingetExecutable
    if (-not $wingetExecutable) {
        throw "winget is not available, so HWiNFO cannot be installed automatically on this host."
    }

    Write-Output "HWiNFO is not installed. Installing REALiX.HWiNFO with winget..."
    & $wingetExecutable install --id REALiX.HWiNFO -e --accept-package-agreements --accept-source-agreements --silent --scope machine
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install REALiX.HWiNFO (exit code $LASTEXITCODE)."
    }
}

function Ensure-HwinfoInstalled {
    $hwinfoExecutable = Get-HwinfoExecutable
    if ($hwinfoExecutable) {
        return $hwinfoExecutable
    }

    Install-HwinfoWithWinget
    $hwinfoExecutable = Get-HwinfoExecutable
    if (-not $hwinfoExecutable) {
        throw "HWiNFO installation completed, but HWiNFO64.EXE was not found in the expected install locations."
    }

    return $hwinfoExecutable
}

function Ensure-PromDapterExecutable {
    $candidate = Join-Path $promDapterInstalledDir "PromDapterSvc.exe"
    if (Test-Path $candidate) {
        return $candidate
    }

    Write-Output "PromDapter is not installed. Downloading PromDapter release..."
    New-Item -ItemType Directory -Force -Path $promDapterDir | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $promDapterDownloadUrl -OutFile $promDapterZip
    Expand-Archive -Path $promDapterZip -DestinationPath $promDapterDir -Force

    $setupExecutable = Get-ChildItem -Path $promDapterDir -Filter "PromDapter-Setup-*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $setupExecutable) {
        throw "PromDapter download completed, but the setup executable was not found after extraction."
    }

    Write-Output "Installing PromDapter..."
    $installProcess = Start-Process -FilePath $setupExecutable.FullName -ArgumentList @('/qn') -PassThru -Wait -WindowStyle Hidden
    if ($installProcess.ExitCode -ne 0) {
        throw "PromDapter installer failed with exit code $($installProcess.ExitCode)."
    }

    if (-not (Test-Path $candidate)) {
        throw "PromDapter installation completed, but PromDapterSvc.exe was not found in $promDapterInstalledDir."
    }

    return $candidate
}

function Start-ManagedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$PidPath,
        [string]$StdOutPath,
        [string]$StdErrPath
    )

    if (Test-Path $PidPath) {
        $existingPidRaw = Get-Content -Path $PidPath -ErrorAction SilentlyContinue | Select-Object -First 1
        $existingPid = 0
        if ([int]::TryParse([string]$existingPidRaw, [ref]$existingPid) -and (Test-ProcessAlive -ProcessId $existingPid)) {
            return $existingPid
        }
        Remove-Item -Force $PidPath -ErrorAction SilentlyContinue
    }

    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WindowStyle Hidden -RedirectStandardOutput $StdOutPath -RedirectStandardError $StdErrPath -PassThru
    Set-Content -Path $PidPath -Value $process.Id
    return $process.Id
}

function Start-GuiProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$PidPath
    )

    if (Test-Path $PidPath) {
        $existingPidRaw = Get-Content -Path $PidPath -ErrorAction SilentlyContinue | Select-Object -First 1
        $existingPid = 0
        if ([int]::TryParse([string]$existingPidRaw, [ref]$existingPid) -and (Test-ProcessAlive -ProcessId $existingPid)) {
            return $existingPid
        }
        Remove-Item -Force $PidPath -ErrorAction SilentlyContinue
    }

    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -PassThru
    Set-Content -Path $PidPath -Value $process.Id
    return $process.Id
}

function Ensure-PromDapterConfig {
    $programDataDir = Join-Path $env:ProgramData "PromDapter"
    $programDataConfig = Join-Path $programDataDir "Prometheusmapping.yaml"
    New-Item -ItemType Directory -Force -Path $programDataDir | Out-Null
    Copy-Item -Path $hwinfoMappingConfig -Destination $programDataConfig -Force
}

function Start-HwinfoBridge {
    New-Item -ItemType Directory -Force -Path $hwinfoStateDir | Out-Null
    New-Item -ItemType Directory -Force -Path $hwinfoVendorDir | Out-Null

    Write-Output "Preparing Windows HWiNFO bridge..."
    $hwinfoExecutable = Ensure-HwinfoInstalled
    $promDapterExecutable = Ensure-PromDapterExecutable
    Ensure-PromDapterConfig

    if (-not (Get-Process -Name "HWiNFO64" -ErrorAction SilentlyContinue)) {
        Start-GuiProcess -FilePath $hwinfoExecutable -ArgumentList @() -PidPath $hwinfoPidPath | Out-Null
        Write-Warning "HWiNFO was started visibly. If it opens the summary screen, click Sensors and enable Shared Memory Support in HWiNFO settings; PromDapter cannot export CPU power until that is done."
    }

    if (-not (Test-HttpEndpoint -Uri "http://127.0.0.1:10445/metrics")) {
        Write-Output "Starting PromDapter on http://127.0.0.1:10445/metrics ..."
        $promDapterService = Get-Service -Name "PromDapterSvc" -ErrorAction SilentlyContinue
        if ($promDapterService) {
            if ($promDapterService.Status -ne "Running") {
                Start-Service -Name "PromDapterSvc" -ErrorAction Stop
            }
        } else {
            Start-ManagedProcess -FilePath $promDapterExecutable -ArgumentList @() -PidPath $promDapterPidPath -StdOutPath $promDapterStdOutPath -StdErrPath $promDapterStdErrPath | Out-Null
        }
    }

    Write-Output "Starting HWiNFO relay into Pushgateway..."
    Start-ManagedProcess -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", ('"' + $hwinfoRelayScript + '"'),
        "-Mode", "push",
        "-SourceUrl", '"http://127.0.0.1:10445/metrics"',
        "-PushUrl", '"http://127.0.0.1:9091/metrics/job/hwinfo/instance/windows-host"',
        "-PushIntervalSeconds", 5
    ) -PidPath $hwinfoRelayPidPath -StdOutPath $hwinfoRelayStdOutPath -StdErrPath $hwinfoRelayStdErrPath | Out-Null

    if (-not (Test-HttpEndpoint -Uri "http://127.0.0.1:10445/metrics")) {
        Write-Warning "PromDapter is not yet serving HWiNFO metrics on http://127.0.0.1:10445/metrics. The relay will keep retrying, but HWiNFO must expose Shared Memory data before CPU power panels will populate."
    }
}

$networkExists = $true
try {
    $null = Invoke-DockerRaw -DockerArgs @("network", "inspect", $networkName) *> $null
    if ($LASTEXITCODE -ne 0) {
        $networkExists = $false
    }
} catch {
    $networkExists = $false
}
if (-not $networkExists) {
    Invoke-Docker @("network", "create", $networkName)
}

Invoke-Docker @("volume", "create", $grafanaVolume)

foreach ($legacyName in @("split-framework-observability-cadvisor", "split-framework-observability-scaphandre")) {
    Remove-ContainerIfPresent -Name $legacyName
}

Invoke-Docker @("build", "-t", $dockerStatsImage, $dockerStatsContextMount)

if (-not (Test-ContainerRunning -Name $keplerName)) {
    Remove-ContainerIfPresent -Name $keplerName
    try {
        Invoke-Docker @(
            "run", "-d",
            "--name", $keplerName,
            "--network", $networkName,
            "--network-alias", "kepler",
            "--privileged",
            "--pid", "host",
            "--gpus", "all",
            "-v", "/sys:/host/sys:ro",
            "-v", "/proc:/host/proc:ro",
            "-v", "/run/nvidia/driver:/run/nvidia/driver:ro",
            "-v", "${keplerMount}:/etc/kepler/config.yaml:ro",
            "-e", "LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib64",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            "-e", "NVIDIA_MIG_MONITOR_DEVICES=all",
            "quay.io/sustainable_computing_io/kepler:latest",
            "--config.file=/etc/kepler/config.yaml"
        )
    } catch {
        Write-Warning "Kepler failed to start. Kepler energy panels will stay empty until host power sensors and GPU runtime are available inside Docker."
    }
}

if (-not (Test-ContainerRunning -Name $prometheusName)) {
    Remove-ContainerIfPresent -Name $prometheusName
    Invoke-Docker @(
        "run", "-d",
        "--name", $prometheusName,
        "--network", $networkName,
        "--network-alias", "prometheus",
        "-v", "${prometheusMount}:/etc/prometheus/prometheus.yml:ro",
        "prom/prometheus:v2.54.1",
        "--config.file=/etc/prometheus/prometheus.yml",
        "--storage.tsdb.path=/prometheus"
    )
}

if (-not (Test-ContainerRunning -Name $pushgatewayName)) {
    Remove-ContainerIfPresent -Name $pushgatewayName
    Invoke-Docker @(
        "run", "-d",
        "--name", $pushgatewayName,
        "--network", $networkName,
        "--network-alias", "pushgateway",
        "-p", "9091:9091",
        "prom/pushgateway:v1.9.0"
    )
}

try {
    Start-HwinfoBridge
} catch {
    Write-Warning "Windows HWiNFO bridge failed to start: $($_.Exception.Message)"
    Write-Warning "Continuing with Prometheus and Grafana startup so the dashboard still opens."
}

if (-not (Test-ContainerRunning -Name $dockerStatsName)) {
    Remove-ContainerIfPresent -Name $dockerStatsName
    Invoke-Docker @(
        "run", "-d",
        "--name", $dockerStatsName,
        "--network", $networkName,
        "--network-alias", "docker-stats",
        "-v", "/var/run/docker.sock:/var/run/docker.sock:ro",
        $dockerStatsImage
    )
}

if (-not (Test-ContainerRunning -Name $grafanaName)) {
    Remove-ContainerIfPresent -Name $grafanaName
    Invoke-Docker @(
        "run", "-d",
        "--name", $grafanaName,
        "--network", $networkName,
        "--network-alias", "grafana",
        "-p", "4000:3000",
        "-e", "GF_SECURITY_ADMIN_USER=admin",
        "-e", "GF_SECURITY_ADMIN_PASSWORD=admin",
        "-e", "GF_USERS_ALLOW_SIGN_UP=false",
        "-v", "${grafanaDatasourceMount}:/etc/grafana/provisioning/datasources/datasources.yml:ro",
        "-v", "${grafanaDashboardsMount}:/etc/grafana/provisioning/dashboards/dashboards.yml:ro",
        "-v", "${grafanaDashboardMount}:/var/lib/grafana/dashboards/split-framework-observability-live.json:ro",
        "-v", "${grafanaVolume}:/var/lib/grafana",
        "grafana/grafana:11.1.4"
    )
}

if (Test-ContainerRunning -Name $grafanaName) {
    Invoke-DockerRaw -DockerArgs @("exec", $grafanaName, "grafana", "cli", "admin", "reset-admin-password", "admin") *> $null
}

Write-Output "Grafana will be available at http://127.0.0.1:4000"
Write-Output "MLflow remains available at http://127.0.0.1:5000"