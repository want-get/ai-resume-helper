"""单机版完整流程自检：门禁 -> 档案 -> 简历 -> 抓岗 -> 薪资 -> 目标岗位 -> 知识库 -> 面试。

    python main.py --port 8900 --no-browser     # 另一个终端先启动
    python scripts/check_app.py --url http://127.0.0.1:8900

会真实联网抓取岗位、并真实调用大模型（如果已配置 Key）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []

RESUME = (
    "张三，3 年 AI 应用开发经验。\n"
    "项目一：企业知识库 RAG 系统。用 FastAPI + Chroma 实现检索增强生成，"
    "把客服回答准确率从 62% 提升到 89%，幻觉率下降 40%。\n"
    "项目二：Agent 工具调用框架。设计 Function Calling 编排层，支持 12 个外部工具并行调用，"
    "平均响应时间从 3.2s 降到 1.1s。\n"
    "技能：Python、FastAPI、asyncio、Chroma、RAG、Prompt Engineering、MySQL。\n"
) * 2


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  ->  {detail}" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main(url: str, role: str, city: str, salary_min: float, salary_max: float,
         skip_crawl: bool) -> int:
    client = httpx.Client(base_url=url, timeout=300)

    # ---------------- 0. 重置，保证可重复运行 ----------------
    if not skip_crawl:
        section("0. 重置档案（保证自检可重复运行）")
        reset = client.delete("/api/v1/profile")
        check("重置档案与专属知识库", reset.status_code == 200,
              f"next_step={reset.json().get('state', {}).get('next_step')}")

    # ---------------- 1. 页面与系统 ----------------
    section("1. 前端页面与系统信息")
    response = client.get("/")
    check("GET / 返回前端页面", response.status_code == 200 and "<title>" in response.text,
          f"HTTP {response.status_code}, {len(response.text)} 字节")
    for asset in ("/static/app.js", "/static/style.css", "/static/favicon.svg"):
        item = client.get(asset)
        check(f"{asset} 可访问", item.status_code == 200 and len(item.content) > 100,
              f"{len(item.content)} 字节")
    info = client.get("/api/v1/system").json()
    check("GET /api/v1/system", "version" in info, f"v{info.get('version')}")
    check("单机模式（无登录/无鉴权）", info.get("single_user_mode") is True)
    check("数据库为 SQLite（无需安装 MySQL）",
          "app.db" in info["paths"]["database"], info["paths"]["database"][-30:])
    print(f"       数据目录：{info['paths']['data_home']}")
    print(f"       大模型　：{info['llm']['provider']} / {info['llm']['model']} "
          f"（已配置={info['llm']['configured']}）")

    check("GET /health", client.get("/health").status_code == 200)

    # ---------------- 2. 必填门禁 ----------------
    section("2. 必填门禁（档案没填全不许继续）")
    state = client.get("/api/v1/profile").json()
    check("空档案 profile_ready=False", state["profile_ready"] is False,
          f"缺：{state['missing_fields']}")

    blocked = client.post("/api/v1/market/crawl", json={})
    check("档案不全时抓岗被拦（HTTP 428）", blocked.status_code == 428,
          f"HTTP {blocked.status_code}：{blocked.json().get('error', '')[:60]}")

    blocked = client.post("/api/v1/interview/start", json={"rounds": 5})
    check("档案不全时面试被拦（HTTP 428）", blocked.status_code == 428,
          f"HTTP {blocked.status_code}")

    # ---------------- 3. 档案与简历 ----------------
    section("3. 填写档案与上传简历")
    state = client.put("/api/v1/profile", json={
        "target_role": role, "expect_city": city,
        "expect_salary_min": salary_min, "expect_salary_max": salary_max,
        "experience_years": 3, "education": "本科",
        "skills": ["Python", "RAG", "Agent", "FastAPI"],
    }).json()
    check("保存档案成功", state["profile"]["target_role"] == role,
          f"{role} / {city} / {salary_min:g}-{salary_max:g}K")
    check("缺失项只剩简历", state["missing_fields"] == ["简历"], str(state["missing_fields"]))
    check("用于匹配的关键字已生成",
          len(state["profile"]["target_keywords"]) > 0,
          str(state["profile"]["target_keywords"]))

    uploaded = client.post("/api/v1/resume/upload",
                           json={"content": RESUME, "filename": "resume.txt"})
    check("上传简历成功", uploaded.status_code == 200,
          f"{uploaded.json().get('chars')} 字" if uploaded.status_code == 200 else uploaded.text[:120])
    state = uploaded.json().get("state", {})
    check("档案已完整 profile_ready=True", state.get("profile_ready") is True)
    print(f"       下一步：{state.get('next_step')}")

    # ---------------- 4. 抓岗与薪资 ----------------
    section("4. 按档案抓取真实岗位并统计薪资")
    report = None
    if skip_crawl:
        print("  （已跳过抓取，复用数据库里的历史结果）")
    else:
        started = time.time()
        response = client.post("/api/v1/market/crawl", json={})
        if response.status_code != 200:
            check("抓取岗位", False, f"HTTP {response.status_code}：{response.text[:150]}")
        else:
            report = response.json()
            check("抓取完成", True,
                  f"{report['total_jobs']} 个岗位 / {report['with_salary']} 个有薪资 / "
                  f"{time.time() - started:.0f}s")
            for source in report.get("sources", []):
                mark = "OK " if source["ok"] else "ERR"
                line = (f"       {mark} {source['label']}: 抓取 {source['fetched']} 条 "
                        f"{source['elapsed_ms']:.0f}ms")
                if source.get("error"):
                    line += f"  错误：{source['error'][:70]}"
                print(line)

    salary = (report or {}).get("salary") or client.get("/api/v1/profile").json()["profile"].get("market_summary", {})
    if salary.get("sample_size"):
        monthly = salary["monthly"]
        check("算出薪资行情", True,
              f"中位 {monthly['median']}K，平均 {monthly['mean']}K，"
              f"P25 {monthly['p25']}K / P75 {monthly['p75']}K，样本 {salary['sample_size']}")
        expectation = salary.get("expectation")
        if expectation:
            check("与期望薪资对比", True,
                  f"{expectation['verdict']}（{expectation['gap_ratio']:+.1f}%）")
            print(f"       建议：{expectation['advice']}")
    else:
        check("算出薪资行情", False, "没有拿到可统计的薪资样本")

    jobs = (report or {}).get("jobs") or client.get("/api/v1/market/jobs").json().get("jobs", [])
    check("拿到岗位列表", len(jobs) > 0, f"{len(jobs)} 条")
    if jobs:
        for job in jobs[:5]:
            print(f"       · {job['title'][:40]:42} | {job.get('city', '')[:10]:10} | "
                  f"{job.get('salary_text') or '—':14} | {job.get('source', '')}")
        check("每条岗位都带来源", all(job.get("source") for job in jobs))
        check("按职位名匹配并有命中标记",
              all(job.get("matched_keywords") for job in jobs[:10]))

    if not jobs:
        print("\n  没有岗位，后续步骤无法继续。")
        return 1

    # ---------------- 5. 目标岗位 ----------------
    section("5. 选定目标岗位")
    target = next((job for job in jobs if job.get("jd_text")), jobs[0])
    response = client.post("/api/v1/market/target", json={"job_key": target["job_key"]})
    check("设为目标岗位", response.status_code == 200,
          response.json().get("target_job", {}).get("title", "")[:50])
    check("提示需要重建知识库", response.json().get("need_rebuild_kb") is True)

    # ---------------- 6. 专属知识库 ----------------
    section("6. 生成用户专属知识库")
    response = client.post("/api/v1/knowledge/personal/build", json={})
    if response.status_code != 200:
        check("生成知识库", False, f"HTTP {response.status_code}：{response.text[:150]}")
    else:
        kb = response.json()
        check("生成知识库", kb.get("ok") is True,
              f"{kb['chunks']} 个文本块（简历 {kb['resume_chunks']} / 目标岗位 "
              f"{kb['target_job_chunks']} / 同类岗位 {kb['similar_job_chunks']} / "
              f"面试题 {kb['question_chunks']}），{kb['seconds']}s")
        for note in kb.get("notes", []):
            print(f"       注意：{note}")
        state = kb.get("state", {})
        check("全部就绪 can_run_interview=True", state.get("can_run_interview") is True,
              f"next_step={state.get('next_step')}")

    stats = client.get("/api/v1/knowledge/personal").json()
    check("个人知识库状态可查询", stats.get("ready") is True, f"{stats.get('chunks')} 块")

    # ---------------- 7. 模拟面试 ----------------
    section("7. 模拟面试（出题基于专属知识库）")
    llm_ready = client.get("/api/v1/profile").json().get("llm_ready")
    if not llm_ready:
        print("  （未配置大模型，跳过面试与优化——请先在前端「模型设置」里填 API Key）")
    else:
        started = time.time()
        response = client.post("/api/v1/interview/start", json={"rounds": 3})
        if response.status_code != 200:
            check("开始面试", False, f"HTTP {response.status_code}：{response.text[:180]}")
        else:
            data = response.json()
            check("生成面试题", bool(data.get("questions")),
                  f"{len(data['questions'])} 字，{time.time() - started:.0f}s，模型={data.get('model')}")
            print("       ----- 题目节选 -----")
            print("       " + data["questions"][:400].replace("\n", "\n       "))
            check("题目带引用来源", len(data.get("sources") or []) > 0,
                  f"{len(data.get('sources') or [])} 条")

            history = [
                {"role": "assistant", "content": data["questions"]},
                {"role": "user", "content":
                    "我在 RAG 项目里用递归切分把 chunk 设为 500 字、重叠 100 字，"
                    "并用向量 + BM25 混合检索加 RRF 融合，召回率提升了约 25%。"},
            ]
            response = client.post("/api/v1/interview/answer",
                                   json={"history": history, "rounds": 3})
            if response.status_code == 200:
                data2 = response.json()
                check("提交回答后给出评价与下一题", bool(data2.get("reply")),
                      f"{len(data2['reply'])} 字")
                memory = data2.get("memory") or {}
                check("返回长上下文指标", "window_messages" in memory,
                      f"窗口 {memory.get('window_messages')} 条 / 摘要 {memory.get('summary_chars')} 字")
            else:
                check("提交回答", False, f"HTTP {response.status_code}：{response.text[:150]}")

            history.append({"role": "assistant", "content": (data2 or {}).get("reply", "")})
            response = client.post("/api/v1/interview/report", json={"history": history})
            check("生成面试报告", response.status_code == 200 and bool(response.json().get("content")),
                  f"{len(response.json().get('content', ''))} 字" if response.status_code == 200 else "")

        # ---------------- 8. 简历优化 ----------------
        section("8. 简历优化（对照目标岗位）")
        started = time.time()
        response = client.post("/api/v1/resume/optimize", json={})
        if response.status_code == 200:
            data = response.json()
            check("生成优化结果", bool(data.get("content")),
                  f"{len(data['content'])} 字，{time.time() - started:.0f}s，"
                  f"可信度 {data.get('confidence', 0) * 100:.0f}%")
            check("带引用来源", len(data.get("sources") or []) > 0)
            print("       ----- 优化节选 -----")
            print("       " + data["content"][:300].replace("\n", "\n       "))
        else:
            check("简历优化", False, f"HTTP {response.status_code}：{response.text[:180]}")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print(f"  通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for item in FAILED:
            print(f"    - {item}")
    print("=" * 78)
    return 1 if FAILED else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单机版完整流程自检")
    parser.add_argument("--url", default="http://127.0.0.1:8900")
    parser.add_argument("--role", default="AI Agent 开发工程师")
    parser.add_argument("--city", default="北京")
    parser.add_argument("--salary-min", type=float, default=25)
    parser.add_argument("--salary-max", type=float, default=40)
    parser.add_argument("--skip-crawl", action="store_true", help="跳过抓取，复用已有数据")
    args = parser.parse_args()
    sys.exit(main(args.url, args.role, args.city, args.salary_min, args.salary_max, args.skip_crawl))
