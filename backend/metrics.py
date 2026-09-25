"""进程内指标采集：并发度、延迟分位、错误率。

用途：
* ``/metrics`` 接口实时展示「当前并发 / 峰值并发 / P95 延迟」，
  这是「支持 100+ 用户并发」这一指标的直接证据。
* 延迟用固定长度的滑动窗口（deque）保存，内存可控，O(n) 统计足够快。
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * ratio))))
    return sorted_values[index]


@dataclass
class Metrics:
    """线程/协程安全的指标容器（单事件循环内使用，无需加锁）。"""

    window_size: int = 5000
    started_at: float = field(default_factory=time.time)

    total_requests: int = 0
    total_errors: int = 0
    in_flight: int = 0
    peak_in_flight: int = 0
    rejected: int = 0

    by_status: dict[int, int] = field(default_factory=dict)
    by_path: dict[str, int] = field(default_factory=dict)

    latencies: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.latencies = deque(maxlen=self.window_size)

    # ------------------------------------------------------------------ #
    def enter(self) -> int:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        return self.in_flight

    def leave(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)

    def record(
        self, path: str, status_code: int, latency_ms: float, in_flight_snapshot: int = 0
    ) -> None:
        self.total_requests += 1
        if status_code >= 400:
            self.total_errors += 1
        self.by_status[status_code] = self.by_status.get(status_code, 0) + 1
        self.by_path[path] = self.by_path.get(path, 0) + 1
        self.latencies.append(latency_ms)
        self.peak_in_flight = max(self.peak_in_flight, in_flight_snapshot)

    def mark_rejected(self) -> None:
        self.rejected += 1
        self.by_status[503] = self.by_status.get(503, 0) + 1

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        values = sorted(self.latencies)
        count = len(values)
        uptime = time.time() - self.started_at
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(self.total_errors / self.total_requests, 4)
            if self.total_requests
            else 0.0,
            "in_flight": self.in_flight,
            "peak_in_flight": self.peak_in_flight,
            "rejected_503": self.rejected,
            "rps": round(self.total_requests / uptime, 2) if uptime > 0 else 0.0,
            "latency_ms": {
                "samples": count,
                "p50": round(_percentile(values, 0.50), 2),
                "p90": round(_percentile(values, 0.90), 2),
                "p95": round(_percentile(values, 0.95), 2),
                "p99": round(_percentile(values, 0.99), 2),
                "max": round(values[-1], 2) if values else 0.0,
                "avg": round(sum(values) / count, 2) if count else 0.0,
            },
            "by_status": {str(k): v for k, v in sorted(self.by_status.items())},
            "top_paths": dict(
                sorted(self.by_path.items(), key=lambda item: item[1], reverse=True)[:10]
            ),
        }


_metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics


class ConcurrencyGate:
    """限制同时在处理的请求数；超时直接返回 503，避免雪崩。

    * 正常情况：立刻拿到令牌，无额外开销。
    * 压测/流量尖峰：在 ``acquire_timeout`` 内排队，超时则快速失败。
    """

    def __init__(self, limit: int, acquire_timeout: float = 15.0) -> None:
        self.limit = max(1, limit)
        self.acquire_timeout = acquire_timeout
        self._semaphore = asyncio.Semaphore(self.limit)
        self.waiting = 0

    async def acquire(self) -> bool:
        self.waiting += 1
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.acquire_timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            self.waiting -= 1

    def release(self) -> None:
        self._semaphore.release()

    def snapshot(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "available": self._semaphore._value,  # noqa: SLF001 - 只读展示
            "waiting": self.waiting,
        }
