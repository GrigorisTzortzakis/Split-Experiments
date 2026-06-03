from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


CONTAINER_ROOT = "/workspace/Split-Framework"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
)
MANAGED_LABEL = "split-framework.launcher=multi-container"
CONTAINER_NAME_PATTERN = re.compile(r"^split-framework-(server|client-\d+|fedserver)$")
NETWORK_NAME_PATTERN = re.compile(r"^split-framework(-\d+-\d+)?-net$")


def _quoted(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _shell_join(command: Sequence[str]) -> str:
    return shlex.join(list(command))


def _print_command(command: Sequence[str]) -> None:
    print(f"$ {_quoted(command)}", flush=True)


def _run(
    command: Sequence[str],
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    _print_command(command)
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed


def _stream(command: Sequence[str], env: dict[str, str] | None = None) -> int:
    _print_command(command)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
    finally:
        if process.stdout is not None:
            process.stdout.close()
    return process.wait()


@dataclass(frozen=True)
class ContainerSpec:
    name: str
    hostname: str
    cpus: str
    memory: str
    memory_swap: str
    shm_size: str
    gpu_device: str | None = None
    mount_docker_socket: bool = False


class DockerMultiContainerLauncher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(args.project_root).resolve()
        self.docker_path = args.docker_path
        self.image = args.image
        self.network_name = "split-framework-net"
        self.container_names: List[str] = []
        self.stopping = False
        self.docker_env = self._resolve_docker_environment()

    def _resolve_docker_environment(self) -> dict[str, str]:
        base_env = os.environ.copy()
        if base_env.get("DOCKER_API_VERSION"):
            return base_env

        return base_env

    def _is_api_mismatch(self, completed: subprocess.CompletedProcess[str]) -> bool:
        combined_output = f"{completed.stdout}\n{completed.stderr}".lower()
        return (
            completed.returncode != 0
            and "requested api version" in combined_output
            and "dockerdesktoplinuxengine" in combined_output
        )

    def _run_docker(self, command: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        completed = _run(command, check=False, env=self.docker_env)
        if completed.returncode == 0 or not self._is_api_mismatch(completed):
            if check and completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, command)
            return completed

        base_env = os.environ.copy()
        for api_version in ("1.43", "1.42", "1.41", "1.40"):
            candidate_env = base_env.copy()
            candidate_env["DOCKER_API_VERSION"] = api_version
            retried = _run(command, check=False, env=candidate_env)
            if retried.returncode == 0:
                self.docker_env = candidate_env
                print(f"Using DOCKER_API_VERSION={api_version} for Docker Desktop compatibility.", flush=True)
                return retried

        if check:
            raise subprocess.CalledProcessError(completed.returncode, command)
        return completed

    def _docker(self, *parts: str) -> List[str]:
        return [self.docker_path, *parts]

    def _list_matching_names(self, command: Sequence[str], pattern: re.Pattern[str]) -> List[str]:
        completed = self._run_docker(command, check=False)
        if completed.returncode != 0:
            return []
        matches: List[str] = []
        for line in completed.stdout.splitlines():
            candidate = line.strip()
            if candidate and pattern.fullmatch(candidate):
                matches.append(candidate)
        return matches

    def _remove_stale_resources(self) -> None:
        stale_containers = self._list_matching_names(
            self._docker("ps", "-a", "--format", "{{.Names}}"),
            CONTAINER_NAME_PATTERN,
        )
        if stale_containers:
            self._run_docker(self._docker("rm", "-f", *stale_containers), check=False)

        stale_networks = self._list_matching_names(
            self._docker("network", "ls", "--format", "{{.Name}}"),
            NETWORK_NAME_PATTERN,
        )
        if stale_networks:
            self._run_docker(self._docker("network", "rm", *stale_networks), check=False)

    def _cleanup(self) -> None:
        if self.container_names:
            self._run_docker(self._docker("rm", "-f", *self.container_names), check=False)
        self._run_docker(self._docker("network", "rm", self.network_name), check=False)

    def _build_specs(self) -> List[ContainerSpec]:
        specs: List[ContainerSpec] = [
            ContainerSpec(
                name="split-framework-server",
                hostname="split-framework-server",
                cpus=self.args.server_cpus,
                memory=self.args.server_memory,
                memory_swap=self.args.server_swap,
                shm_size=self.args.shm_size,
                gpu_device="0" if self.args.device == "gpu" else None,
                mount_docker_socket=True,
            )
        ]
        for index in range(1, self.args.clients + 1):
            specs.append(
                ContainerSpec(
                    name=f"split-framework-client-{index}",
                    hostname=f"split-framework-client-{index}",
                    cpus=self.args.client_cpus,
                    memory=self.args.client_memory,
                    memory_swap=self.args.client_swap,
                    shm_size=self.args.shm_size,
                )
            )
        if self.args.with_fed_server:
            specs.append(
                ContainerSpec(
                    name="split-framework-fedserver",
                    hostname="split-framework-fedserver",
                    cpus=self.args.fed_server_cpus,
                    memory=self.args.fed_server_memory,
                    memory_swap=self.args.fed_server_swap,
                    shm_size=self.args.shm_size,
                )
            )
        return specs

    def _start_containers(self, specs: Iterable[ContainerSpec]) -> None:
        self._remove_stale_resources()
        self._run_docker(self._docker("network", "create", self.network_name))
        for spec in specs:
            command = [
                self.docker_path,
                "run",
                "-d",
                "--rm",
                "--name",
                spec.name,
                "--hostname",
                spec.hostname,
                "--network",
                self.network_name,
                "--network-alias",
                spec.hostname,
                "--label",
                MANAGED_LABEL,
                "--cpus",
                spec.cpus,
                "--memory",
                spec.memory,
                "--memory-swap",
                spec.memory_swap,
                "--shm-size",
                spec.shm_size,
                "-v",
                f"{self.project_root}:{CONTAINER_ROOT}",
                "-w",
                CONTAINER_ROOT,
                "--init",
            ]
            for env_var in THREAD_ENV_VARS:
                command.extend(["-e", env_var])
            if spec.gpu_device is not None:
                command.extend(["--gpus", f"device={spec.gpu_device}"])
            if spec.mount_docker_socket:
                command.extend(["-v", "/var/run/docker.sock:/var/run/docker.sock"])
            command.extend([self.image, "tail", "-f", "/dev/null"])
            self._run_docker(command)
            self.container_names.append(spec.name)

    def _launch_mpi(self, specs: Sequence[ContainerSpec]) -> int:
        launcher = specs[0]
        host_list = ",".join(f"{spec.hostname}:1" for spec in specs)
        mpi_command = [
            "mpirun",
            "--allow-run-as-root",
            "-np",
            str(len(specs)),
            "-H",
            host_list,
            "--mca",
            "plm_rsh_agent",
            "docker-exec",
            "python",
            *self.args.experiment_args,
        ]
        docker_exec_command = self._docker(
            "exec",
            launcher.name,
            "sh",
            "-lc",
            _shell_join(mpi_command),
        )
        return _stream(docker_exec_command, env=self.docker_env)

    def run(self) -> int:
        specs = self._build_specs()
        try:
            self._start_containers(specs)
            return self._launch_mpi(specs)
        finally:
            self._cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Split-Framework with one Docker container per MPI rank.")
    parser.add_argument("--docker-path", required=True)
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

    launcher = DockerMultiContainerLauncher(args)

    def _handle_signal(_signum: int, _frame: object) -> None:
        launcher._cleanup()
        raise SystemExit(1)

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), _handle_signal)

    return launcher.run()


if __name__ == "__main__":
    raise SystemExit(main())