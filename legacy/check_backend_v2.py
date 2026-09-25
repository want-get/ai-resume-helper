"""后端端到端自检（不消耗真实 API 额度）。

    python scripts/check_backend.py

默认强制开启离线 Mock 模式（``LLM_MOCK_MODE=true``），
因此可以在没有 DeepSeek API Key 的情况下验证完整链路：
数据库、知识库、RAG、防幻觉引用校验、Function Calling、多轮对话与长上下文摘要。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 必须在导入 config 之前设置
os.environ.setdefault("LLM_MOCK_MODE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from backend.api import app  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


LONG_ANSWER = (
    "我在上一家公司负责订单中台的后端开发，主导了从单体到微服务的拆分，"
    "把订单创建接口的 P95 延迟从 480ms 降到 120ms，QPS 从 300 提升到 1500。"
    "技术上使用 FastAPI 异步框架，MySQL 分库分表，Redis 做热点缓存，"
    "并用 asyncio 信号量控制对下游库存服务的并发，避免雪崩。"
    "期间还遇到了缓存与数据库不一致的问题，最终通过先更新库再删除缓存加延迟双删解决。"
)


def main() -> int:
    with TestClient(app) as client:
        # ---------------- 基础 ----------------
        section("1. 基础接口 / 数据库 / 知识库")
        r = client.get("/health")
        check("GET /health 返回 200", r.status_code == 200, str(r.status_code))
        health = r.json()
        check(
            "数据库已连接",
            health["database"]["backend"] in ("mysql", "sqlite"),
            f"{health['database']['backend']} fallback={health['database']['fallback']}",
        )
        check(
            "知识库已建库",
            all(item["chunks"] > 0 for item in health["knowledge_base"]),
            str([(i["collection"], i["chunks"]) for i in health["knowledge_base"]]),
        )
        check("大模型处于 Mock 模式", health["llm"]["mock_mode"] is True)
        print(f"     数据库：{health['database']['url']}")
        print(f"     并发配置：{health['concurrency']}")

        r = client.get("/metrics")
        check("GET /metrics 返回 200", r.status_code == 200)

        r = client.get("/api/v1/stats")
        check("GET /api/v1/stats 返回 200", r.status_code == 200, str(r.json()))

        # ---------------- RAG ----------------
        section("2. RAG 知识库检索（向量 + BM25 混合）")
        r = client.post(
            "/api/v1/knowledge/search",
            json={"query": "缓存穿透 缓存雪崩 怎么解决", "top_k": 3},
        )
        check("POST /api/v1/knowledge/search 返回 200", r.status_code == 200)
        data = r.json()
        check("检索到结果", len(data["chunks"]) > 0, f"{len(data['chunks'])} 条 / {data['elapsed_ms']}ms")
        check(
            "命中缓存相关题目",
            any("缓存" in c["text"] for c in data["chunks"]),
            data["chunks"][0]["label"] if data["chunks"] else "",
        )
        check(
            "返回融合来源标记",
            all(c["source"] in ("vector", "bm25", "hybrid") for c in data["chunks"]),
            str([c["source"] for c in data["chunks"]]),
        )

        section("3. RAG 问答 + 防幻觉引用校验")
        r = client.post(
            "/api/v1/rag/ask",
            json={"question": "缓存穿透和缓存雪崩分别怎么解决？", "role_type": "tech"},
        )
        check("POST /api/v1/rag/ask 返回 200", r.status_code == 200, r.text[:200])
        answer = r.json()
        check("返回了回答", bool(answer["content"]))
        check("带回了引用来源", len(answer["sources"]) > 0, f"{len(answer['sources'])} 条来源")
        check(
            "来源带编号与可读标签",
            all(s["index"] >= 1 and s["label"] for s in answer["sources"]),
            answer["sources"][0]["label"] if answer["sources"] else "",
        )
        check("返回引用校验报告", answer["citation_report"] is not None)
        check("返回可信度分数", 0.0 <= answer["confidence"] <= 1.0, f"{answer['confidence']}")
        print(f"     引用校验：{answer['citation_report']}")

        r = client.post("/api/v1/rag/ask", json={"question": "今天天气怎么样？适合穿什么衣服？"})
        refused = r.json()
        check(
            "知识库无资料时拒答（不自由发挥）",
            refused["refused"] is True,
            f"refused={refused['refused']} confidence={refused['confidence']}",
        )

        # ---------------- Function Calling ----------------
        section("4. Function Calling：工具直调")
        r = client.post(
            "/api/v1/tools/salary",
            json={"role": "Python后端开发", "city": "杭州", "level": "中级"},
        )
        check("POST /api/v1/tools/salary 返回 200", r.status_code == 200)
        salary = r.json()
        check("薪资查询命中数据", salary.get("found") is True, str(salary.get("matched_role")))
        check(
            "返回分位数据与样本量",
            "percentiles" in salary and salary.get("sample_size", 0) > 0,
            str(salary.get("percentiles")),
        )
        check("标注数据来源", bool(salary.get("source")), salary.get("source", ""))
        print(f"     薪资结果：{salary.get('matched_role')} {salary.get('city')} "
              f"{salary.get('percentiles')} {salary.get('unit')}")

        r = client.post(
            "/api/v1/tools/jobs",
            json={"keywords": "Python 后端", "sources": "local", "limit": 5},
        )
        check("POST /api/v1/tools/jobs 返回 200", r.status_code == 200)
        jobs = r.json()
        check("岗位搜索命中结果", jobs.get("found") is True and len(jobs.get("jobs", [])) > 0,
              f"{len(jobs.get('jobs', []))} 条")
        check(
            "只按职位名称匹配且标注命中关键字",
            all(job.get("matched_keywords") for job in jobs.get("jobs", [])),
            str([job["matched_keywords"] for job in jobs.get("jobs", [])][:2]),
        )
        if jobs.get("jobs"):
            print(f"     岗位：{jobs['jobs'][0]['title']} @ {jobs['jobs'][0]['company']} "
                  f"{jobs['jobs'][0]['salary']}")

        r = client.post(
            "/api/v1/tools/jobs",
            json={"keywords": "星野科技", "sources": "local", "limit": 5},
        )
        noise = r.json()
        check(
            "公司名不应命中（关键字只匹配职位名）",
            noise.get("found") is False,
            f"found={noise.get('found')} jobs={len(noise.get('jobs', []))}",
        )

        section("4.1 岗位数据源（真实公开接口）")
        sources_info = client.get("/api/v1/tools/job-sources").json()
        check(
            "列出全部数据源并说明用途",
            len(sources_info["available"]) >= 5,
            "、".join(item["key"] for item in sources_info["available"]),
        )
        r = client.post(
            "/api/v1/tools/jobs",
            json={"keywords": "python backend AI Agent", "sources": "jobicy,remotive", "limit": 5},
        )
        live = r.json()
        ok_sources = [item["source"] for item in live.get("sources", []) if item["ok"]]
        check("公开招聘接口可访问", bool(ok_sources), f"可用数据源：{ok_sources}")
        if live.get("jobs"):
            check(
                "抓到真实岗位并带原文链接",
                all(job.get("url") for job in live["jobs"]),
                f"{len(live['jobs'])} 条，首条：{live['jobs'][0]['title']}",
            )
            print(f"     实时岗位：{live['jobs'][0]['title']} @ {live['jobs'][0].get('company')}"
                  f"（{live['jobs'][0].get('source')}）")
        else:
            print(f"     公开数据源本次未命中关键字（{ok_sources} 已连通）")

        section("4.2 在线设置 API Key（免改 .env / 免重启）")
        llm = client.get("/api/v1/settings/llm").json()
        check("GET 设置接口返回状态", "configured" in llm and "key_masked" in llm)
        check(
            "接口不回显明文 Key",
            "deepseek_api_key" not in llm and "api_key" not in llm,
            f"key_masked={llm.get('key_masked')!r}",
        )
        original_key = llm.get("key_masked")
        r = client.put(
            "/api/v1/settings/llm",
            json={"api_key": "sk-selftest-not-real-000000", "persist": False, "verify": False},
        )
        updated = r.json()
        check("可以在线更新 Key（不写 .env）",
              r.status_code == 200 and updated["status"]["configured"] is True,
              f"mock_mode={updated['status'].get('mock_mode')}")
        check("更新后立即生效（Mock 模式关闭）", updated["status"]["mock_mode"] is False)
        check("不写回 .env 时 persisted 为空", updated["persisted_env_keys"] == [])

        r = client.post("/api/v1/tools/call",
                        json={"name": "search_knowledge_base", "arguments": {"query": "GIL", "top_k": 1}})
        check("伪 Key 下工具仍可用（不依赖模型）", r.status_code == 200)

        client.delete("/api/v1/settings/llm/key", params={"persist": "false"})
        restored = client.get("/api/v1/settings/llm").json()
        check("清除后回到 Mock 模式", restored["mock_mode"] is True)
        print(f"     原始 Key 掩码：{original_key!r}（未被写回 .env，已还原运行时状态）")

        r = client.post(
            "/api/v1/tools/call",
            json={"name": "search_knowledge_base", "arguments": {"query": "GIL 多线程", "top_k": 2}},
        )
        check("POST /api/v1/tools/call 返回 200", r.status_code == 200)
        check("知识库工具被正确执行", r.json()["result"]["ok"] is True)

        section("5. Function Calling：模型自主决策")
        r = client.post(
            "/api/v1/tools/ask",
            json={"question": "杭州 Python 后端开发 3 年经验的薪资大概是多少？", "use_tools": True},
        )
        check("POST /api/v1/tools/ask 返回 200", r.status_code == 200, r.text[:200])
        agent = r.json()
        check("模型发起了工具调用", len(agent["tool_calls"]) > 0,
              str([t["name"] for t in agent["tool_calls"]]))
        check(
            "工具调用成功并回传结果",
            all(t["ok"] for t in agent["tool_calls"]) and bool(agent["tool_calls"]),
            str([(t["name"], t["ok"]) for t in agent["tool_calls"]]),
        )
        check("生成了最终回答", bool(agent["content"]))
        print(f"     工具链：{[t['name'] for t in agent['tool_calls']]}")

        # ---------------- 简历 ----------------
        section("6. 简历相关功能（RAG 增强）")
        r = client.post(
            "/api/v1/resumes",
            json={"filename": "张三-后端开发.pdf", "content": LONG_ANSWER * 3},
        )
        check("POST /api/v1/resumes 返回 200", r.status_code == 200)
        resume_id = r.json()["resume_id"]

        r = client.get(f"/api/v1/resumes/{resume_id}")
        check("GET /api/v1/resumes/{id} 返回 200", r.status_code == 200)

        for path, label in (
            ("/api/v1/resume/questions", "面试题生成"),
            ("/api/v1/resume/optimize", "简历优化"),
            ("/api/v1/resume/score", "简历评分"),
        ):
            r = client.post(path, json={"resume_id": resume_id, "role_type": "tech"})
            body = r.json() if r.status_code == 200 else {}
            check(
                f"{label} 返回 200 且有内容",
                r.status_code == 200 and bool(body.get("content")),
                f"{r.status_code} sources={len(body.get('sources', []))}",
            )

        # ---------------- 多轮对话 ----------------
        section("7. 多轮对话 + 长上下文（滑动窗口 + 滚动摘要）")
        r = client.post(
            "/api/v1/sessions",
            json={
                "mode": "mock_interview",
                "role_type": "tech",
                "title": "后端岗位模拟面试",
                "resume_text": LONG_ANSWER * 2,
            },
        )
        check("创建会话返回 200", r.status_code == 200, r.text[:200])
        session_id = r.json()["session_id"]

        r = client.post(f"/api/v1/sessions/{session_id}/start")
        check("面试开场返回 200", r.status_code == 200, r.text[:200])
        check("开场有面试官提问", bool(r.json()["reply"]["content"]))

        memory_reports = []
        for i in range(10):
            r = client.post(
                f"/api/v1/sessions/{session_id}/messages",
                json={"content": f"第{i + 1}轮回答：{LONG_ANSWER}"},
            )
            if r.status_code != 200:
                check(f"第 {i + 1} 轮对话返回 200", False, r.text[:200])
                break
            memory_reports.append(r.json()["memory"])
        else:
            check("连续 10 轮对话全部成功", True)

        r = client.get(f"/api/v1/sessions/{session_id}")
        session = r.json()
        check("消息已持久化", session["message_count"] >= 20, f"{session['message_count']} 条")

        r = client.get(f"/api/v1/sessions/{session_id}/context")
        check("GET context 返回 200", r.status_code == 200)
        ctx = r.json()
        stats = ctx["stats"]
        check(
            "滑动窗口只保留最近若干条原文",
            0 < stats["window_messages"] < stats["total_messages"],
            f"窗口 {stats['window_messages']} / 总计 {stats['total_messages']} 条",
        )
        check(
            "早期对话已被压缩为摘要",
            stats["summary_chars"] > 0 and stats["summarized_messages"] > 0,
            f"摘要 {stats['summary_chars']} 字，覆盖 {stats['summarized_messages']} 条消息",
        )
        check(
            "上下文整体被压缩",
            stats["compressed_ratio"] > 0,
            f"压缩比 {stats['compressed_ratio']}（历史 {stats['history_chars']} 字 → "
            f"窗口 {stats['window_chars']} 字 + 摘要 {stats['summary_chars']} 字）",
        )
        check("摘要内容非空", bool(ctx["summary"].strip()))
        last = memory_reports[-1] if memory_reports else {}
        check(
            "滚动摘要在多轮中触发过",
            any(m and m.get("summary_updated") for m in memory_reports),
            f"最后一轮 stats={last}",
        )

        r = client.post(f"/api/v1/sessions/{session_id}/report")
        check("生成面试报告返回 200", r.status_code == 200 and bool(r.json().get("content")))

        section("8. 通用知识库多轮对话（带工具）")
        r = client.post(
            "/api/v1/sessions",
            json={"mode": "rag_chat", "role_type": "tech", "title": "知识库问答"},
        )
        chat_id = r.json()["session_id"]
        r = client.post(
            f"/api/v1/sessions/{chat_id}/messages",
            json={"content": "面试官常问的缓存问题有哪些？", "use_tools": True},
        )
        check("rag_chat 对话返回 200", r.status_code == 200, r.text[:200])
        body = r.json()
        check("返回引用来源", len(body["result"]["sources"]) > 0,
              f"{len(body['result']['sources'])} 条")
        check("返回长上下文指标", body["memory"] is not None)

        r = client.get("/api/v1/sessions")
        check("会话列表返回 200", r.status_code == 200 and len(r.json()["sessions"]) >= 2)

        # ---------------- 收尾 ----------------
        section("9. 指标汇总")
        r = client.get("/metrics")
        snap = r.json()["app"]
        print(f"     总请求 {snap['total_requests']}，错误 {snap['total_errors']}，"
              f"错误率 {snap['error_rate']}")
        print(f"     延迟 P50={snap['latency_ms']['p50']}ms P95={snap['latency_ms']['p95']}ms "
              f"P99={snap['latency_ms']['p99']}ms")
        print(f"     峰值在飞请求 {snap['peak_in_flight']}")

    print("\n" + "=" * 78)
    print(f"  通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("  失败项：")
        for item in FAILED:
            print(f"    - {item}")
        print("=" * 78)
        return 1
    print("  ✅ 后端端到端自检全部通过")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
