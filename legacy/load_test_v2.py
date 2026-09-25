"""并发压测脚本：验证后端能支撑 100+ 用户并发。

    python scripts/load_test.py                      # 进程内起服务 + 120 并发用户
    python scripts/load_test.py --users 200 --rounds 3
    python scripts/load_test.py --url http://127.0.0.1:8000   # 压测已启动的服务

默认在**进程内**启动 uvicorn（真实 HTTP 协议栈，不是 TestClient），
用 ``httpx.AsyncClient`` 开足连接池并发打流量，最后汇总：

* 成功率 / 吞吐（RPS）/ P50 P90 P95 P99 延迟
* 服务端自报的峰值在飞请求数（来自 /metrics）
* 每个接口的调用量

默认走离线 Mock 模式（不配置 DEEPSEEK_API_KEY 时自动如此），
所以压测不会消耗真实 API 额度；它验证的是**本服务的并发调度能力**
（事件循环、并发闸门、数据库连接池、Chroma 线程池），
真实模型场景下瓶颈在上游配额，由 MAX_CONCURRENT_LLM 控制。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PORT = 8765


# ---------------------------------------------------------------------- #
# 流量编排
# ---------------------------------------------------------------------- #
QUERIES = [
    "缓存穿透和缓存雪崩分别怎么解决？",
    "GIL 对多线程有什么影响？",
    "如何提升新用户 7 日留存？",
    "面试官常问的缓存问题有哪些？",
    "请介绍一下 RAG 的完整链路",
    "asyncio 事件循环是怎么工作的？",
    "MySQL 索引什么情况下会失效？",
    "Function Calling 的原理是什么？",
]

JOB_KEYWORDS = ["Python", "大模型", "前端", "算法", "产品经理", "运营"]


def build_scenarios() -> list[tuple[str, str, dict]]:
    """(名称, 方法, 参数) 列表，模拟真实用户行为分布。"""
    return [
        ("GET /health", "GET", {"path": "/health"}),
        ("POST /api/v1/knowledge/search", "POST", {
            "path": "/api/v1/knowledge/search",
            "json": {"query": "{query}", "top_k": 3},
        }),
        ("POST /api/v1/rag/ask", "POST", {
            "path": "/api/v1/rag/ask",
            "json": {"question": "{query}", "role_type": "tech"},
        }),
        ("POST /api/v1/tools/ask", "POST", {
            "path": "/api/v1/tools/ask",
            "json": {"question": "杭州 Python 后端 3 年经验薪资多少？", "use_tools": True},
        }),
        ("POST /api/v1/tools/jobs", "POST", {
            "path": "/api/v1/tools/jobs",
            "json": {"keyword": "{keyword}", "limit": 3},
        }),
        ("POST /api/v1/tools/salary", "POST", {
            "path": "/api/v1/tools/salary",
            "json": {"role": "Python后端开发工程师", "city": "杭州"},
        }),
        ("GET /api/v1/stats", "GET", {"path": "/api/v1/stats"}),
    ]


# 权重：检索与问答占大头，贴近真实使用
SCENARIO_WEIGHTS = [1, 3, 3, 2, 2, 1, 1]


class Stats:
    def __init__(self) -> None:
        self.latencies: list[float] = []
        self.ok = 0
        self.failed = 0
        self.status: dict[int, int] = {}
        self.by_path: dict[str, list[float]] = {}
        self.errors: list[str] = []

    def record(self, name: str, status: int, latency_ms: float) -> None:
        self.status[status] = self.status.get(status, 0) + 1
        if 200 <= status < 400:
            self.ok += 1
        else:
            self.failed += 1
            if len(self.errors) < 10:
                self.errors.append(f"{name} -> HTTP {status}")
        self.latencies.append(latency_ms)
        self.by_path.setdefault(name, []).append(latency_ms)

    @staticmethod
    def _pct(values: list[float], ratio: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio)))]

    def summary(self, wall_seconds: float) -> dict[str, object]:
        total = self.ok + self.failed
        return {
            "total_requests": total,
            "ok": self.ok,
            "failed": self.failed,
            "success_rate": round(self.ok / total * 100, 2) if total else 0.0,
            "wall_seconds": round(wall_seconds, 2),
            "rps": round(total / wall_seconds, 1) if wall_seconds > 0 else 0.0,
            "p50_ms": round(self._pct(self.latencies, 0.50), 2),
            "p90_ms": round(self._pct(self.latencies, 0.90), 2),
            "p95_ms": round(self._pct(self.latencies, 0.95), 2),
            "p99_ms": round(self._pct(self.latencies, 0.99), 2),
            "max_ms": round(max(self.latencies), 2) if self.latencies else 0.0,
            "avg_ms": round(statistics.mean(self.latencies), 2) if self.latencies else 0.0,
            "status": dict(sorted(self.status.items())),
            "errors": self.errors,
        }


async def virtual_user(
    client,
    user_id: int,
    rounds: int,
    stats: Stats,
    scenarios: list[tuple[str, str, dict]],
) -> None:
    """一个虚拟用户：连续发起 rounds 次请求。"""
    for _ in range(rounds):
        name, method, template = random.choices(scenarios, weights=SCENARIO_WEIGHTS, k=1)[0]
        kwargs = {k: v for k, v in template.items() if k != "path"}
        payload = kwargs.get("json")
        if payload:
            kwargs["json"] = _render(payload)
        started = time.perf_counter()
        try:
            response = await client.request(method, template["path"], **kwargs)
            stats.record(name, response.status_code, (time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001
            stats.record(name, 0, (time.perf_counter() - started) * 1000)
            if len(stats.errors) < 10:
                stats.errors.append(f"{name} -> {type(exc).__name__}: {exc}")


def _render(payload: dict) -> dict:
    out = {}
    for key, value in payload.items():
        if isinstance(value, str) and "{query}" in value:
            out[key] = value.replace("{query}", random.choice(QUERIES))
        elif isinstance(value, str) and "{keyword}" in value:
            out[key] = value.replace("{keyword}", random.choice(JOB_KEYWORDS))
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------- #
async def start_in_process_server(port: int):
    import uvicorn

    os.environ.setdefault("LLM_MOCK_MODE", "true")
    config = uvicorn.Config(
        "backend.api:app", host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.1)
    if not server.started:
        raise RuntimeError("服务启动超时")
    return server, task


async def run(args: argparse.Namespace) -> int:
    import httpx

    server = server_task = None
    base_url = args.url

    if not base_url:
        print(f"正在进程内启动 uvicorn（127.0.0.1:{args.port}）...")
        server, server_task = await start_in_process_server(args.port)
        base_url = f"http://127.0.0.1:{args.port}"
        await asyncio.sleep(1.0)  # 等待 lifespan 完成建库/建知识库

    print("=" * 78)
    print(f"  目标：{base_url}")
    print(f"  并发用户：{args.users}    每个用户请求数：{args.rounds}")
    print(f"  总请求量：≈ {args.users * args.rounds}")
    print("=" * 78)

    limits = httpx.Limits(
        max_connections=args.users + 50, max_keepalive_connections=args.users + 50
    )
    timeout = httpx.Timeout(args.timeout)
    stats = Stats()
    scenarios = build_scenarios()

    try:
        async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=timeout) as client:
            # 健康检查
            health = (await client.get("/health")).json()
            print(f"  服务版本：{health['version']}    数据库：{health['database']['backend']}"
                  f"（回退={health['database']['fallback']}）")
            print(f"  大模型：mock_mode={health['llm']['mock_mode']}  "
                  f"并发上限={health['concurrency']['limit']}")
            print("=" * 78)

            before = (await client.get("/metrics")).json()["app"]

            started = time.perf_counter()
            await asyncio.gather(
                *[
                    virtual_user(client, i, args.rounds, stats, scenarios)
                    for i in range(args.users)
                ]
            )
            wall = time.perf_counter() - started

            after = (await client.get("/metrics")).json()["app"]
            db_stats = (await client.get("/api/v1/stats")).json()
    finally:
        if server is not None:
            server.should_exit = True
            if server_task is not None:
                await server_task

    summary = stats.summary(wall)
    print("\n" + "=" * 78)
    print("  压测结果")
    print("=" * 78)
    print(f"  并发用户数　　: {args.users}")
    print(f"  总请求 / 成功 : {summary['total_requests']} / {summary['ok']}"
          f"（失败 {summary['failed']}）")
    print(f"  成功率　　　　: {summary['success_rate']}%")
    print(f"  墙钟耗时　　　: {summary['wall_seconds']}s")
    print(f"  吞吐　　　　　: {summary['rps']} req/s")
    print(f"  延迟 P50/P90　: {summary['p50_ms']}ms / {summary['p90_ms']}ms")
    print(f"  延迟 P95/P99　: {summary['p95_ms']}ms / {summary['p99_ms']}ms")
    print(f"  最大延迟　　　: {summary['max_ms']}ms")
    print(f"  HTTP 状态分布 : {summary['status']}")
    print(f"  服务端峰值并发: {after['peak_in_flight']}（本轮新增 "
          f"{after['peak_in_flight'] - before['peak_in_flight']}）")
    print(f"  服务端 503 拒绝: {after['rejected_503']}")
    print(f"  数据库近 1 小时: {db_stats.get('recent_requests', '（接口未返回，可能被限流）')}")

    print("\n  分接口延迟 (P95 / 调用量)：")
    for name, values in sorted(stats.by_path.items(), key=lambda kv: -len(kv[1])):
        print(f"    {name:38} {Stats._pct(values, 0.95):8.2f}ms   n={len(values)}")

    if summary["errors"]:
        print("\n  错误样例：")
        for err in summary["errors"]:
            print(f"    - {err}")

    print("\n" + "=" * 78)
    ok = summary["failed"] == 0 and args.users >= 100
    if ok:
        print(f"  ✅ 通过：{args.users} 并发用户全部请求成功，成功率 100%")
    elif summary["failed"] == 0:
        print(f"  ⚠️  成功率为 100%，但并发用户数 {args.users} < 100，"
              f"请用 --users 120 复跑以验证 100+ 并发")
    else:
        print(f"  ❌ 存在失败请求：{summary['failed']} 条")
    print("=" * 78)
    return 0 if summary["failed"] == 0 else 1


def main() -> None:
    import logging

    # 压测时 httpx 的逐条请求日志会把结果淹没，压掉
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ai_resume_helper").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="FastAPI 后端并发压测")
    parser.add_argument("--users", type=int, default=120, help="并发用户数（默认 120）")
    parser.add_argument("--rounds", type=int, default=3, help="每个用户的请求数（默认 3）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="进程内服务的端口")
    parser.add_argument("--url", default=None, help="压测已有服务，例如 http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60.0, help="单请求超时（秒）")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
