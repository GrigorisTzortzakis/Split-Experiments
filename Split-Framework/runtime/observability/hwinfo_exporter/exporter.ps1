param(
    [ValidateSet("push")]
    [string]$Mode = "push",

    [string]$SourceUrl = "http://127.0.0.1:10445/metrics",

    [Parameter(Mandatory = $true)]
    [string]$PushUrl,

    [int]$PushIntervalSeconds = 5
)

$ErrorActionPreference = "Stop"

if ($PushIntervalSeconds -lt 1) {
    throw "PushIntervalSeconds must be at least 1."
}

function Get-HwinfoMetricPayload {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $SourceUrl -TimeoutSec 10
    $lines = [System.Collections.Generic.List[string]]::new()

    foreach ($line in ($response.Content -split "`r?`n")) {
        if ($line -match '^hwi_.*_w\{') {
            $lines.Add($line)
        }
    }

    if (-not ($lines | Where-Object { $_ -match '^hwi_' })) {
        throw "No hwi_ metrics were found at $SourceUrl. Check that HWiNFO is running with Shared Memory enabled and PromDapter can read it."
    }

    return (($lines -join "`n") + "`n")
}

function Get-BridgeStatusPayload {
    param(
        [bool]$IsHealthy,
        [string]$Message
    )

    $statusValue = if ($IsHealthy) { 1 } else { 0 }
    $safeMessage = $Message -replace '\\', '\\\\' -replace '"', '\\"'

    return @(
        '# HELP hwinfo_bridge_up Whether the HWiNFO relay could fetch HWiNFO-backed metrics from PromDapter.',
        '# TYPE hwinfo_bridge_up gauge',
        "hwinfo_bridge_up $statusValue",
        '# HELP hwinfo_bridge_status Information label describing the last relay state.',
        '# TYPE hwinfo_bridge_status gauge',
        ('hwinfo_bridge_status{message="' + $safeMessage + '"} 1')
    ) -join "`n"
}

while ($true) {
    try {
        $payload = Get-HwinfoMetricPayload
        $statusPayload = Get-BridgeStatusPayload -IsHealthy $true -Message "ok"
        $body = $payload + $statusPayload + "`n"
        Invoke-WebRequest -UseBasicParsing -Method Put -Uri $PushUrl -ContentType 'text/plain; version=0.0.4' -Body $body | Out-Null
    } catch {
        $statusPayload = Get-BridgeStatusPayload -IsHealthy $false -Message $_.Exception.Message
        Invoke-WebRequest -UseBasicParsing -Method Put -Uri $PushUrl -ContentType 'text/plain; version=0.0.4' -Body ($statusPayload + "`n") | Out-Null
    }

    Start-Sleep -Seconds $PushIntervalSeconds
}