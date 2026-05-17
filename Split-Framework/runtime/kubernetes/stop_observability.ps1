param(
    [string]$ProjectRoot = ""
)

Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match '^kubectl(\.exe)?$' -and
        $_.CommandLine -like '*port-forward*service/grafana 4000:3000*'
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

if ($ProjectRoot) {
    $kustomizePath = Join-Path $ProjectRoot "runtime/kubernetes"
    if (Test-Path $kustomizePath) {
        kubectl delete -k $kustomizePath --ignore-not-found=true | Out-Null
    }
}
