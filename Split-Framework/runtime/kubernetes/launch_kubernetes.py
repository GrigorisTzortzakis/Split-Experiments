from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


CONTAINER_ROOT = "/workspace/Split-Framework"
RUN_NAMESPACE = "split-framework-runs"
SERVICE_ACCOUNT_NAME = "split-framework-launcher"
THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "SPLIT_FRAMEWORK_K8S_NAMESPACE": RUN_NAMESPACE,
}


def _quoted(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _print_command(command: Sequence[str]) -> None:
    print(f"$ {_quoted(command)}", flush=True)


def _run(command: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    _print_command(command)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed


def _stream(command: Sequence[str]) -> int:
    _print_command(command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
    finally:
        if process.stdout is not None:
            process.stdout.close()
    return process.wait()


def _linux_host_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1].lstrip("/")
    return f"/run/desktop/mnt/host/{drive}/{tail}"


def _normalize_windows_path(path: str) -> str:
    normalized = path.replace("\\", "/").rstrip("/")
    if len(normalized) >= 2 and normalized[1] == ":":
        normalized = normalized[0].lower() + normalized[1:]
    return normalized


def _k8s_memory_quantity(value: str) -> str:
    raw = str(value).strip()
    lowered = raw.lower()
    if lowered.endswith("gi") or lowered.endswith("mi") or lowered.endswith("ki"):
        return raw
    if lowered.endswith("g"):
        return f"{raw[:-1]}Gi"
    if lowered.endswith("m"):
        return f"{raw[:-1]}Mi"
    return raw


@dataclass(frozen=True)
class PodSpec:
    name: str
    role: str
    cpu_limit: str
    memory_limit: str
    shm_size: str
    gpu_count: int = 0


class KubernetesMultiPodLauncher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.kubectl_path = args.kubectl_path
        self.project_root = Path(args.project_root).resolve()
        self.image = args.image
        self.run_id = f"split-framework-{int(time.time())}-{os.getpid()}"
        self.pod_names: List[str] = []
        self.current_context = self._kubectl_output("config", "current-context").strip()
        self.k3d_node_containers = self._discover_k3d_node_containers()
        self.node_project_root = self._resolve_node_project_root()

    def _kubectl(self, *parts: str) -> List[str]:
        return [self.kubectl_path, *parts]

    def _kubectl_output(self, *parts: str) -> str:
        return _run(self._kubectl(*parts)).stdout

    def _docker(self, *parts: str) -> List[str]:
        return ["docker", *parts]

    def _docker_output(self, *parts: str) -> str:
        return _run(self._docker(*parts)).stdout

    def _discover_k3d_node_containers(self) -> List[str]:
        if not self.current_context.startswith("k3d-"):
            return []
        cluster_name = self.current_context
        output = self._docker_output("ps", "--format", "{{.Names}}")
        node_names: List[str] = []
        for line in output.splitlines():
            name = line.strip()
            if not name.startswith(f"{cluster_name}-"):
                continue
            suffix = name[len(cluster_name) + 1 :]
            if suffix.startswith("server-") or suffix.startswith("agent-"):
                node_names.append(name)
        return node_names

    def _resolve_node_project_root(self) -> str:
        if not self.k3d_node_containers:
            return _linux_host_path(self.project_root)
        target_source = _normalize_windows_path(str(self.project_root))
        inspect_output = self._docker_output("inspect", self.k3d_node_containers[0])
        data = json.loads(inspect_output)
        mounts = data[0].get("Mounts", []) if data else []
        for mount in mounts:
            source = _normalize_windows_path(str(mount.get("Source", "")))
            if source == target_source:
                destination = str(mount.get("Destination", "")).strip()
                if destination:
                    return destination
        return _linux_host_path(self.project_root)

    def _import_image_into_k3d(self) -> None:
        if not self.k3d_node_containers:
            return
        for node in self.k3d_node_containers:
            save_command = self._docker("save", self.image)
            import_command = self._docker("exec", "-i", node, "ctr", "-n", "k8s.io", "images", "import", "-")
            _print_command(save_command)
            save_process = subprocess.Popen(save_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert save_process.stdout is not None
            try:
                _print_command(import_command)
                completed = subprocess.run(import_command, stdin=save_process.stdout, text=False, capture_output=True)
            finally:
                save_process.stdout.close()
            stderr = save_process.stderr.read().decode(errors="replace") if save_process.stderr is not None else ""
            if save_process.stderr is not None:
                save_process.stderr.close()
            save_return_code = save_process.wait()
            if completed.stdout:
                print(completed.stdout.decode(errors="replace"), end="", flush=True)
            if completed.stderr:
                print(completed.stderr.decode(errors="replace"), end="", file=sys.stderr, flush=True)
            if stderr:
                print(stderr, end="", file=sys.stderr, flush=True)
            if save_return_code != 0:
                raise subprocess.CalledProcessError(save_return_code, save_command)
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, import_command)

    def _ensure_gpu_capacity(self) -> None:
        if self.args.device != "gpu":
            return
        output = self._kubectl_output("get", "nodes", "-o", "json")
        data = json.loads(output)
        total_gpus = 0
        for item in data.get("items", []):
            allocatable = item.get("status", {}).get("allocatable", {})
            try:
                total_gpus += int(str(allocatable.get("nvidia.com/gpu", "0")))
            except ValueError:
                continue
        if total_gpus < 1:
            raise RuntimeError(
                "Kubernetes backend requested device=gpu, but the active cluster exposes no allocatable nvidia.com/gpu resources. "
                "Fix the cluster GPU runtime or switch the launcher device to cpu."
            )

    def _raise_pod_failure(self, pod_name: str) -> None:
        describe = _run(self._kubectl("describe", "pod", pod_name, "-n", RUN_NAMESPACE), check=False)
        raise RuntimeError(
            f"Pod {pod_name} did not become Ready. See describe output above for the exact Kubernetes failure."
        )

    def _apply_manifest(self, manifest: Dict[str, object]) -> None:
        command = self._kubectl("apply", "-f", "-")
        _print_command(command)
        completed = subprocess.run(command, input=json.dumps(manifest), text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)

    def _ensure_namespace_and_rbac(self) -> None:
        self._ensure_gpu_capacity()
        self._import_image_into_k3d()
        _run(self._kubectl("create", "namespace", RUN_NAMESPACE), check=False)
        manifest = {
            "apiVersion": "v1",
            "kind": "List",
            "items": [
                {
                    "apiVersion": "v1",
                    "kind": "ServiceAccount",
                    "metadata": {"name": SERVICE_ACCOUNT_NAME, "namespace": RUN_NAMESPACE},
                },
                {
                    "apiVersion": "rbac.authorization.k8s.io/v1",
                    "kind": "Role",
                    "metadata": {"name": "split-framework-launcher", "namespace": RUN_NAMESPACE},
                    "rules": [
                        {
                            "apiGroups": [""],
                            "resources": ["pods", "pods/exec", "pods/log"],
                            "verbs": ["get", "list", "watch", "create", "delete"],
                        }
                    ],
                },
                {
                    "apiVersion": "rbac.authorization.k8s.io/v1",
                    "kind": "RoleBinding",
                    "metadata": {"name": "split-framework-launcher", "namespace": RUN_NAMESPACE},
                    "subjects": [
                        {"kind": "ServiceAccount", "name": SERVICE_ACCOUNT_NAME, "namespace": RUN_NAMESPACE}
                    ],
                    "roleRef": {
                        "apiGroup": "rbac.authorization.k8s.io",
                        "kind": "Role",
                        "name": "split-framework-launcher",
                    },
                },
            ],
        }
        self._apply_manifest(manifest)

    def _build_specs(self) -> List[PodSpec]:
        specs = [
            PodSpec(
                name=f"{self.run_id}-server",
                role="server",
                cpu_limit=self.args.server_cpus,
                memory_limit=self.args.server_memory,
                shm_size=self.args.shm_size,
                gpu_count=1 if self.args.device == "gpu" else 0,
            )
        ]
        for index in range(1, self.args.clients + 1):
            specs.append(
                PodSpec(
                    name=f"{self.run_id}-client-{index}",
                    role="client",
                    cpu_limit=self.args.client_cpus,
                    memory_limit=self.args.client_memory,
                    shm_size=self.args.shm_size,
                )
            )
        if self.args.with_fed_server:
            specs.append(
                PodSpec(
                    name=f"{self.run_id}-fedserver",
                    role="fedserver",
                    cpu_limit=self.args.fed_server_cpus,
                    memory_limit=self.args.fed_server_memory,
                    shm_size=self.args.shm_size,
                )
            )
        return specs

    def _pod_manifest(self, spec: PodSpec) -> Dict[str, object]:
        memory_limit = _k8s_memory_quantity(spec.memory_limit)
        shm_size = _k8s_memory_quantity(spec.shm_size)
        resources: Dict[str, Dict[str, str]] = {
            "requests": {"cpu": spec.cpu_limit, "memory": memory_limit},
            "limits": {"cpu": spec.cpu_limit, "memory": memory_limit},
        }
        if spec.gpu_count:
            resources["limits"]["nvidia.com/gpu"] = str(spec.gpu_count)
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": spec.name,
                "namespace": RUN_NAMESPACE,
                "labels": {
                    "app": "split-framework",
                    "split-framework.run": self.run_id,
                    "split-framework.role": spec.role,
                },
            },
            "spec": {
                "serviceAccountName": SERVICE_ACCOUNT_NAME,
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": "runner",
                        "image": self.image,
                        "imagePullPolicy": "IfNotPresent",
                        "workingDir": CONTAINER_ROOT,
                        "command": ["tail", "-f", "/dev/null"],
                        "env": [{"name": key, "value": value} for key, value in THREAD_ENV_VARS.items()],
                        "resources": resources,
                        "volumeMounts": [
                            {"name": "project-root", "mountPath": CONTAINER_ROOT},
                            {"name": "devshm", "mountPath": "/dev/shm"},
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "project-root",
                        "hostPath": {"path": self.node_project_root, "type": "Directory"},
                    },
                    {
                        "name": "devshm",
                        "emptyDir": {"medium": "Memory", "sizeLimit": shm_size},
                    },
                ],
            },
        }

    def _start_pods(self, specs: Sequence[PodSpec]) -> None:
        self._ensure_namespace_and_rbac()
        for spec in specs:
            self._apply_manifest(self._pod_manifest(spec))
            self.pod_names.append(spec.name)
        for pod_name in self.pod_names:
            result = _run(
                self._kubectl("wait", "--for=condition=Ready", f"pod/{pod_name}", "-n", RUN_NAMESPACE, "--timeout=180s"),
                check=False,
            )
            if result.returncode != 0:
                self._raise_pod_failure(pod_name)

    def _launch_mpi(self, specs: Sequence[PodSpec]) -> int:
        launcher = specs[0]
        host_list = ",".join(f"{spec.name}:1" for spec in specs)
        mpi_command = [
            "mpirun",
            "--allow-run-as-root",
            "-np",
            str(len(specs)),
            "-H",
            host_list,
            "--mca",
            "plm_rsh_agent",
            "kubectl-exec",
            "python",
            *self.args.experiment_args,
        ]
        return _stream(
            self._kubectl(
                "exec",
                "-n",
                RUN_NAMESPACE,
                launcher.name,
                "--",
                "sh",
                "-lc",
                shlex.join(mpi_command),
            )
        )

    def _cleanup(self) -> None:
        for pod_name in self.pod_names:
            _run(self._kubectl("delete", "pod", pod_name, "-n", RUN_NAMESPACE, "--ignore-not-found=true"), check=False)

    def run(self) -> int:
        specs = self._build_specs()
        try:
            self._start_pods(specs)
            return self._launch_mpi(specs)
        finally:
            self._cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Split-Framework with one Kubernetes pod per MPI rank.")
    parser.add_argument("--kubectl-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--with-fed-server", action="store_true")
    parser.add_argument("--server-cpus", required=True)
    parser.add_argument("--server-memory", required=True)
    parser.add_argument("--server-swap", required=True)
    parser.add_argument("--client-cpus", required=True)
    parser.add_argument("--client-memory", required=True)
    parser.add_argument("--client-swap", required=True)
    parser.add_argument("--fed-server-cpus", required=True)
    parser.add_argument("--fed-server-memory", required=True)
    parser.add_argument("--fed-server-swap", required=True)
    parser.add_argument("--shm-size", required=True)
    parser.add_argument("experiment_args", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.experiment_args and args.experiment_args[0] == "--":
        args.experiment_args = args.experiment_args[1:]
    launcher = KubernetesMultiPodLauncher(args)

    def _handle_signal(_signum: int, _frame: object) -> None:
        launcher._cleanup()
        raise SystemExit(1)

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), _handle_signal)
    return launcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
