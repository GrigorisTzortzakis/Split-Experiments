import os
import threading
import time

import docker
from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sum_network_bytes(networks, key):
    total = 0.0
    if not isinstance(networks, dict):
        return total

    for stats in networks.values():
        if isinstance(stats, dict):
            total += _safe_float(stats.get(key, 0.0))
    return total


def _sum_block_bytes(entries, operation):
    total = 0.0
    if not isinstance(entries, list):
        return total

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("op", "")).lower() != operation:
            continue
        total += _safe_float(entry.get("value", 0.0))
    return total


def _cpu_percent(stats):
    cpu_stats = stats.get("cpu_stats", {}) or {}
    precpu_stats = stats.get("precpu_stats", {}) or {}

    cpu_usage = ((cpu_stats.get("cpu_usage") or {}).get("total_usage"))
    precpu_usage = ((precpu_stats.get("cpu_usage") or {}).get("total_usage"))
    system_usage = cpu_stats.get("system_cpu_usage")
    presystem_usage = precpu_stats.get("system_cpu_usage")

    if cpu_usage is None or precpu_usage is None or system_usage is None or presystem_usage is None:
        return 0.0

    cpu_delta = _safe_float(cpu_usage) - _safe_float(precpu_usage)
    system_delta = _safe_float(system_usage) - _safe_float(presystem_usage)
    if cpu_delta <= 0 or system_delta <= 0:
        return 0.0

    online_cpus = cpu_stats.get("online_cpus")
    if not online_cpus:
        per_cpu = (cpu_stats.get("cpu_usage") or {}).get("percpu_usage") or []
        online_cpus = max(len(per_cpu), 1)

    return (cpu_delta / system_delta) * float(online_cpus) * 100.0


class DockerStatsCollector:
    def __init__(self):
        socket_path = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
        self.client = docker.DockerClient(base_url=socket_path)
        self.api = self.client.api
        self.refresh_interval = float(os.environ.get("EXPORTER_REFRESH_SECONDS", "5"))
        self._lock = threading.Lock()
        self._snapshot = None
        self._snapshot_error = None
        self._worker = threading.Thread(target=self._refresh_loop, daemon=True)
        self._worker.start()

    def _refresh_loop(self):
        while True:
            try:
                snapshot = self._collect_snapshot()
                with self._lock:
                    self._snapshot = snapshot
                    self._snapshot_error = None
            except Exception as exc:
                with self._lock:
                    self._snapshot_error = str(exc)
            time.sleep(self.refresh_interval)

    def _collect_snapshot(self):
        containers = self.client.containers.list(all=False)
        snapshot = []
        for container in containers:
            attrs = container.attrs or {}
            try:
                stats = self.api.stats(container.id, stream=False, one_shot=True)
            except Exception:
                continue

            image_tags = container.image.tags if container.image is not None else []
            image = ((attrs.get("Config") or {}).get("Image")) or (image_tags[0] if image_tags else "")
            state = ((attrs.get("State") or {}).get("Status")) or container.status or "unknown"
            snapshot.append(
                {
                    "container_id": container.id,
                    "container_name": container.name,
                    "image": image,
                    "state": state,
                    "stats": stats,
                }
            )

        return snapshot

    def collect(self):
        labels = ["container_id", "container_name", "image", "state"]
        cpu_usage_seconds = CounterMetricFamily(
            "docker_container_cpu_usage_seconds_total",
            "Total CPU time consumed by the container in seconds.",
            labels=labels,
        )
        cpu_percent = GaugeMetricFamily(
            "docker_container_cpu_percent",
            "Approximate container CPU usage percent based on the latest Docker stats sample.",
            labels=labels,
        )
        memory_usage = GaugeMetricFamily(
            "docker_container_memory_usage_bytes",
            "Current container memory usage in bytes.",
            labels=labels,
        )
        memory_limit = GaugeMetricFamily(
            "docker_container_memory_limit_bytes",
            "Current container memory limit in bytes.",
            labels=labels,
        )
        memory_percent = GaugeMetricFamily(
            "docker_container_memory_percent",
            "Current container memory usage as a percentage of the configured limit.",
            labels=labels,
        )
        network_rx = CounterMetricFamily(
            "docker_container_network_receive_bytes_total",
            "Total bytes received by the container across all interfaces.",
            labels=labels,
        )
        network_tx = CounterMetricFamily(
            "docker_container_network_transmit_bytes_total",
            "Total bytes transmitted by the container across all interfaces.",
            labels=labels,
        )
        block_read = CounterMetricFamily(
            "docker_container_block_read_bytes_total",
            "Total bytes read by the container from block devices.",
            labels=labels,
        )
        block_write = CounterMetricFamily(
            "docker_container_block_write_bytes_total",
            "Total bytes written by the container to block devices.",
            labels=labels,
        )

        with self._lock:
            containers = self._snapshot
            snapshot_error = self._snapshot_error

        if containers is None:
            exc = snapshot_error or "docker stats snapshot is not ready yet"
            error_metric = GaugeMetricFamily(
                "docker_stats_exporter_up",
                "Whether the exporter can query Docker successfully.",
                labels=["error"],
            )
            error_metric.add_metric([str(exc)], 0)
            yield error_metric
            return

        up_metric = GaugeMetricFamily(
            "docker_stats_exporter_up",
            "Whether the exporter can query Docker successfully.",
        )
        up_metric.add_metric([], 1)
        yield up_metric

        for container in containers:
            stats = container.get("stats") or {}
            container_id = container.get("container_id", "")
            container_name = container.get("container_name", "")
            image = container.get("image", "")
            state = container.get("state", "unknown")
            label_values = [container_id, container_name, image, state]

            total_cpu_ns = _safe_float((((stats.get("cpu_stats") or {}).get("cpu_usage") or {}).get("total_usage")))
            cpu_usage_seconds.add_metric(label_values, total_cpu_ns / 1_000_000_000.0)
            cpu_percent.add_metric(label_values, _cpu_percent(stats))

            memory_stats = stats.get("memory_stats") or {}
            usage_bytes = _safe_float(memory_stats.get("usage", 0.0))
            limit_bytes = _safe_float(memory_stats.get("limit", 0.0))
            memory_usage.add_metric(label_values, usage_bytes)
            memory_limit.add_metric(label_values, limit_bytes)
            memory_percent.add_metric(label_values, (usage_bytes / limit_bytes * 100.0) if limit_bytes > 0 else 0.0)

            networks = stats.get("networks") or {}
            network_rx.add_metric(label_values, _sum_network_bytes(networks, "rx_bytes"))
            network_tx.add_metric(label_values, _sum_network_bytes(networks, "tx_bytes"))

            blkio_stats = (stats.get("blkio_stats") or {}).get("io_service_bytes_recursive") or []
            block_read.add_metric(label_values, _sum_block_bytes(blkio_stats, "read"))
            block_write.add_metric(label_values, _sum_block_bytes(blkio_stats, "write"))

        yield cpu_usage_seconds
        yield cpu_percent
        yield memory_usage
        yield memory_limit
        yield memory_percent
        yield network_rx
        yield network_tx
        yield block_read
        yield block_write


if __name__ == "__main__":
    REGISTRY.register(DockerStatsCollector())
    port = int(os.environ.get("EXPORTER_PORT", "9417"))
    start_http_server(port)
    while True:
        time.sleep(3600)