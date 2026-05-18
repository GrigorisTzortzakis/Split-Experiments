param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
$namespace = "split-framework-observability"
$kustomizePath = Join-Path $ProjectRoot "runtime/kubernetes"

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    throw "kubectl was not found on PATH. Enable Docker Desktop Kubernetes or install kubectl first."
}

Set-Location $ProjectRoot
try {
    $currentContext = (kubectl config current-context 2>$null).Trim()
} catch {
    $currentContext = ""
}

if (-not $currentContext) {
    throw "Kubernetes is not configured on this machine. Enable Docker Desktop Kubernetes and make sure kubectl has a current context before starting observability."
}

try {
    kubectl cluster-info | Out-Null
} catch {
    throw "Kubernetes is not reachable from kubectl. Start Docker Desktop Kubernetes and verify the current context before starting observability."
}

kubectl apply -k $kustomizePath | Out-String | Write-Output
kubectl rollout status -n $namespace deployment/prometheus --timeout=180s | Out-String | Write-Output
kubectl rollout status -n $namespace deployment/grafana --timeout=180s | Out-String | Write-Output
kubectl rollout status -n $namespace daemonset/cadvisor --timeout=180s | Out-String | Write-Output
try {
    kubectl rollout status -n $namespace daemonset/kepler --timeout=120s | Out-String | Write-Output
} catch {
    Write-Warning "Kepler did not become ready. Energy panels will stay empty until the node exposes the required power counters."
}

Write-Output "Grafana will be available at http://127.0.0.1:4000"
Write-Output "MLflow remains available at http://127.0.0.1:5000"
kubectl port-forward -n $namespace service/grafana 4000:3000
