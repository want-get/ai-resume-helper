"""命令行版主程序（客户端）。

    python server.py      # 先启动后端
    python main.py        # 再运行本客户端

本版本是**后端的一个客户端**：所有 AI 能力（RAG 检索、Function Calling、
多轮对话、长上下文压缩）都在 FastAPI 后端里，这里只负责交互与展示。
原来的四大功能（简历优化 / 生成面试题 / 模拟面试 / 简历评分）全部保留，
并新增知识库问答、薪资与岗位查询（Function Calling）、知识库运维。
"""

from __future__ import annotations

import os
import sys

from client import (
    BackendClient,
    BackendError,
    format_confidence,
    format_memory,
    format_sources,
    format_target_job,
    format_tool_calls,
)
from prompts import NON_TECH, ROLE_LABELS, TECH

MENU = """
{line}
    AI 面试与简历助手 v2.0（FastAPI + DeepSeek + RAG + Function Calling）
    当前岗位：{role}    后端：{backend}
{line}
  1. 简历优化                 （RAG 增强：按目标岗位 / 岗位 JD 优化）
  2. 生成面试题               （RAG 增强：参考题库同类题）
  3. 模拟面试                 （多轮对话 + 滑动窗口 + 滚动摘要）
  4. 简历评分                 （RAG 增强：按目标岗位 / 岗位 JD 打分）
  5. 知识库问答               （RAG + 引用来源 + 防幻觉校验）
  6. 薪资 / 岗位查询           （Function Calling：真实公开招聘数据）
  7. 搜索并选定目标岗位        （选好后 1/2/3/4 都会围绕它展开）
  8. 知识库状态 / 重建
  9. 查看会话长上下文压缩情况
 10. 模型设置（在线填 API Key，免重启）
 11. 切换岗位类型
  0. 退出
{line}
"""

LINE = "=" * 72


def choose_role_type() -> str:
    print("\n请选择岗位类型：")
    print("  1. 技术岗")
    print("  2. 非技术岗")
    while True:
        choice = input("请输入（1-2）: ").strip()
        if choice == "1":
            return TECH
        if choice == "2":
            return NON_TECH
        print("无效选择，请输入 1 或 2")


# ---------------------------------------------------------------------- #
# 输入助手
# ---------------------------------------------------------------------- #
def read_resume(client: BackendClient) -> str | None:
    """上传 PDF 简历（也可以直接粘贴文本），返回 resume_id。"""
    print("\n请提供简历：")
    print("  1. 输入 PDF 文件路径（推荐，后端用 pdfplumber 解析）")
    print("  2. 直接粘贴简历文本")
    print("  0. 返回菜单")
    choice = input("请选择（0-2）: ").strip()

    if choice == "0":
        return None

    if choice == "1":
        path = input("请输入 PDF 完整路径：").strip().strip('"').strip("'")
        if not path:
            print("路径不能为空")
            return None
        if not os.path.exists(path):
            print(f"❌ 文件不存在：{path}")
            return None
        print("\n正在上传并解析 PDF...")
        try:
            result = client.upload_resume(path)
        except BackendError as exc:
            print(f"❌ {exc}")
            return None
        print(f"✅ 解析成功（{result['chars']} 字），resume_id={result['resume_id']}")
        return result["resume_id"]

    if choice == "2":
        print("请输入简历文本，输入一行单独的点号 . 结束：")
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == ".":
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if len(text) < 30:
            print("❌ 文本太短（至少 30 字）")
            return None
        result = client.create_resume("pasted-resume.txt", text)
        print(f"✅ 已保存，resume_id={result['resume_id']}")
        return result["resume_id"]

    print("无效选择")
    return None


def save_result(default_name: str) -> None:
    """把结果保存到文件（沿用原版交互）。"""
    answer = input("是否保存结果到文件？(y/n): ").strip().lower()
    if answer != "y":
        return
    filename = input(f"请输入文件名（默认：{default_name}）: ").strip() or default_name
    if not filename.endswith(".txt"):
        filename += ".txt"
    return filename


def show_result(result: dict, default_filename: str) -> None:
    """统一展示 AI 结果 + 防幻觉指标 + 引用来源。"""
    print("\n" + "-" * 72)
    print(result.get("content", ""))
    print("-" * 72)
    print(f"📊 {format_confidence(result)}")

    sources = result.get("sources") or []
    if sources:
        print("\n📚 引用来源（可在知识库中核对）：")
        print(format_sources(sources))

    if result.get("mock"):
        print("\n⚠️  当前为离线 Mock 模式（未配置 DEEPSEEK_API_KEY），内容为桩响应。")

    filename = save_result(default_filename)
    if filename:
        try:
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write(result.get("content", ""))
                if sources:
                    handle.write("\n\n=== 引用来源 ===\n")
                    handle.write(format_sources(sources, preview=1000))
            print(f"✅ 已保存到 {filename}")
        except OSError as exc:
            print(f"❌ 保存失败：{exc}")


# ---------------------------------------------------------------------- #
# 功能
# ---------------------------------------------------------------------- #
def feature_search_and_pick_job(client: BackendClient, state: dict) -> None:
    """搜索真实岗位并选定「目标岗位」，后续功能都会围绕它展开。"""
    print(f"\n{LINE}\n          搜索并选定目标岗位\n{LINE}")
    print("关键字只匹配【职位名称】，多个关键字用空格或逗号分隔，之间是「或」的关系。")
    print("例如：python backend、AI、Agent、产品经理\n")

    current = state.get("target_job")
    if current:
        print(f"当前目标岗位：{current.get('title')}"
              + (f" @ {current['company']}" if current.get("company") else ""))
        print()

    try:
        info = client.job_sources()
        print("可用数据源：")
        for source in info["available"]:
            print(f"  · {source['key']}：{source['label']}")
        print(f"  （{info.get('note', '')}）")
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    keywords = input("\n请输入搜索关键字（回车取消）> ").strip()
    if not keywords:
        return

    print("\n正在从公开招聘接口抓取岗位，请稍候...")
    try:
        result = client.jobs(keywords=keywords, limit=10)
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    jobs = result.get("jobs") or []
    print("\n各数据源情况：")
    for item in result.get("sources") or []:
        mark = "✅" if item.get("ok") else "❌"
        print(f"  {mark} {item['source']}：抓取 {item.get('fetched')} 条，命中 {item.get('matched')} 条"
              + (f"（{item.get('error')}）" if item.get("error") else ""))
    for note in result.get("notes") or []:
        print(f"  ⚠️  {note}")

    if not jobs:
        print("\n没有职位名匹配到这些关键字的岗位，试试更通用的词（如 python / backend / AI）。")
        return

    print(f"\n找到 {len(jobs)} 个岗位（按命中关键字数量排序）：\n")
    for index, job in enumerate(jobs, start=1):
        salary = job.get("salary") or "未提供"
        print(f"  [{index}] {job.get('title')}")
        print(f"      {job.get('company') or '未提供'} | {job.get('location') or '未提供'} | {salary}")
        if job.get("tags"):
            print(f"      标签：{'、'.join(str(t) for t in job['tags'][:8])}")
        print(f"      来源：{job.get('source')}    链接：{job.get('url') or '无'}")
        print()

    raw = input("输入编号把它设为目标岗位（回车跳过，0 清除当前目标岗位）> ").strip()
    if raw == "0":
        state["target_job"] = None
        print("已清除目标岗位")
        return
    if not raw.isdigit() or not 1 <= int(raw) <= len(jobs):
        print("已跳过")
        return

    state["target_job"] = jobs[int(raw) - 1]
    print(f"\n✅ 已设为目标岗位：{state['target_job'].get('title')}")
    print("   之后「简历优化 / 面试题 / 模拟面试 / 简历评分」都会围绕它展开。")


def feature_llm_settings(client: BackendClient) -> None:
    """在线填 API Key，立即生效，不用改 .env 也不用重启后端。"""
    print(f"\n{LINE}\n          模型设置（在线填 API Key）\n{LINE}")
    try:
        status = client.llm_settings()
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    print(f"当前状态：{'已配置' if status['configured'] else '未配置'}"
          f"（Mock 模式：{'是' if status['mock_mode'] else '否'}）")
    print(f"模型　　　：{status['model']}")
    print(f"接口地址　：{status['base_url']}")
    if status.get("key_masked"):
        print(f"当前 Key　：{status['key_masked']}")
    if not status.get("runtime_editable", True):
        print("⚠️  后端已关闭在线修改配置（ALLOW_RUNTIME_SETTINGS=false），请改 .env 后重启")
        return

    print("\n来源：https://platform.deepseek.com/ → API Keys")
    print("直接回车可跳过；输入 clear 可清除当前 Key；输入 verify 可验证当前 Key")
    raw = input("请输入新的 API Key > ").strip()

    if not raw:
        print("已跳过")
        return
    if raw.lower() == "verify":
        print("正在验证...")
        try:
            check = client.verify_llm()
            print(("✅ 可用：" + str(check.get("reply_preview"))) if check.get("ok")
                  else f"❌ {check.get('error')}")
        except BackendError as exc:
            print(f"❌ {exc}")
        return
    if raw.lower() == "clear":
        try:
            client.clear_llm_key(persist=True)
            print("✅ 已清除 Key，回到 Mock 模式")
        except BackendError as exc:
            print(f"❌ {exc}")
        return

    print("\n正在保存并验证（会真实调用一次模型）...")
    try:
        outcome = client.update_llm_settings(api_key=raw, persist=True, verify=True)
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    new_status = outcome.get("status", {})
    print(f"✅ 已生效：{new_status.get('model')}（Key {new_status.get('key_masked')}）")
    if outcome.get("persisted_env_keys"):
        print(f"   已写回 .env：{'、'.join(outcome['persisted_env_keys'])}")
    check = outcome.get("verify")
    if check:
        print(("✅ 验证通过：" + str(check.get("reply_preview"))) if check.get("ok")
              else f"❌ 验证失败：{check.get('error')}")


def feature_resume(client: BackendClient, role_type: str, feature: str, title: str,
                   state: dict) -> None:
    print(f"\n{LINE}\n          {title}\n{LINE}")
    job_context = format_target_job(state.get("target_job")) or None
    if job_context:
        print(f"🎯 目标岗位：{state['target_job'].get('title')}")
    else:
        print("（未选定目标岗位，只依据简历与内置示例库；建议先用菜单 7 选一个）")
    print()

    resume_id = read_resume(client)
    if resume_id is None:
        return

    print("\nAI 正在处理（RAG 检索 + 生成 + 引用校验），请稍候...")
    try:
        method = {
            "optimize": client.optimize_resume,
            "questions": client.interview_questions,
            "score": client.score_resume,
        }[feature]
        result = method(role_type=role_type, resume_id=resume_id, job_context=job_context)
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    default_names = {
        "optimize": "optimized_resume.txt",
        "questions": "interview_questions.txt",
        "score": "resume_score.txt",
    }
    show_result(result, default_names[feature])


def feature_mock_interview(client: BackendClient, role_type: str, state: dict) -> None:
    print(f"\n{LINE}\n          AI 模拟面试（多轮对话）\n{LINE}")
    print("提示：面试过程中可输入 quit 提前结束；输入 context 查看长上下文压缩情况\n")

    job_context = format_target_job(state.get("target_job")) or None
    job_title = (state.get("target_job") or {}).get("title", "")
    job_company = (state.get("target_job") or {}).get("company", "")
    if job_title:
        print(f"🎯 面试将围绕目标岗位：{job_title}"
              + (f" @ {job_company}" if job_company else ""))
    else:
        print("（未选定目标岗位；建议先用菜单 7 选一个，面试会更贴近真实岗位）")
    print()

    resume_id = read_resume(client)
    if resume_id is None:
        return

    try:
        session = client.create_session(
            mode="mock_interview",
            role_type=role_type,
            title=f"{ROLE_LABELS[role_type]}模拟面试",
            resume_id=resume_id,
            job_title=job_title,
            job_company=job_company,
            job_context=job_context,
        )
        session_id = session["session_id"]
        opening = client.start_session(session_id)
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    print("\n" + "-" * 72)
    print(f"【面试官】（第 1 题）\n{opening['reply']['content']}")
    print("-" * 72)

    round_number = 1
    while True:
        try:
            answer = input("\n【你】（quit 结束 / context 查看上下文）: ").strip()
        except EOFError:
            break

        if answer.lower() == "quit":
            break
        if answer.lower() == "context":
            try:
                context = client.session_context(session_id)
                print(f"\n🧠 {format_memory(context.get('stats'))}")
                print(f"摘要内容：{context.get('summary', '')[:400]}")
            except BackendError as exc:
                print(f"❌ {exc}")
            continue
        if not answer:
            print("请输入你的回答！")
            continue

        try:
            payload = client.send_message(session_id, answer)
        except BackendError as exc:
            print(f"❌ {exc}")
            break

        round_number += 1
        print("\n" + "-" * 72)
        print(f"【面试官】（第 {round_number} 题）\n{payload['reply']['content']}")
        print("-" * 72)
        print(f"🧠 {format_memory(payload.get('memory'))}")

    print("\n面试结束，正在生成结构化评估报告...")
    try:
        report = client.session_report(session_id)
    except BackendError as exc:
        print(f"❌ {exc}")
        return
    show_result(report, "interview_report.txt")


def feature_rag_chat(client: BackendClient, role_type: str) -> None:
    print(f"\n{LINE}\n          知识库问答（RAG + 引用来源）\n{LINE}")
    print("输入 quit 退出；每轮都会先检索题库与岗位 JD，再作答并给出引用。\n")

    while True:
        try:
            question = input("你的问题> ").strip()
        except EOFError:
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        try:
            result = client.rag_ask(question, role_type=role_type)
        except BackendError as exc:
            print(f"❌ {exc}")
            continue

        print("\n" + result.get("content", ""))
        print("-" * 72)
        print(f"📊 {format_confidence(result)}")
        if result.get("refused"):
            print("🛡️  防幻觉：知识库无相关资料，已拒绝自由发挥。")
        for warning in result.get("warnings") or []:
            print(f"⚠️  {warning}")
        sources = result.get("sources") or []
        if sources:
            print("\n📚 引用来源：")
            print(format_sources(sources))


def feature_tools(client: BackendClient, role_type: str, state: dict) -> None:
    print(f"\n{LINE}\n          薪资 / 岗位查询（Function Calling）\n{LINE}")
    print("AI 会自主决定调用哪些工具（薪资查询 / 岗位搜索 / 知识库检索）。")
    print("岗位搜索的关键字只匹配【职位名称】，例如：python backend、AI、Agent\n")
    try:
        tools = client.list_tools()
        print("已注册工具：")
        for tool in tools:
            print(f"  · {tool['name']}：{tool['description'][:60]}...")
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    while True:
        try:
            question = input("\n你的问题（quit 退出 / jobs 直接搜岗位）> ").strip()
        except EOFError:
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        if question.lower() == "jobs":
            feature_search_and_pick_job(client, state)
            continue

        try:
            result = client.tools_ask(question, role_type=role_type)
        except BackendError as exc:
            print(f"❌ {exc}")
            continue

        calls = result.get("tool_calls") or []
        if calls:
            print("\n🔧 工具调用链：")
            print(format_tool_calls(calls))
            # 岗位搜索结果容易被淹没，这里额外把岗位列表摊平展示
            for call in calls:
                data = call.get("result") or {}
                if call.get("name") == "search_jobs" and data.get("jobs"):
                    print(f"\n📋 {data.get('source', '')}")
                    for index, job in enumerate(data["jobs"], start=1):
                        print(f"  [{index}] {job.get('title')} @ {job.get('company') or '未提供'}"
                              f" | {job.get('location') or '未提供'} | {job.get('salary') or '未提供'}")
                        if job.get("url"):
                            print(f"      {job['url']}")
                    print("      （想把它设为目标岗位：先按 quit 退出，再用菜单 7）")
        print("\n" + str(result.get("content", "")))
        for warning in result.get("warnings") or []:
            print(f"⚠️  {warning}")


def feature_knowledge(client: BackendClient) -> None:
    print(f"\n{LINE}\n          知识库状态 / 重建\n{LINE}")
    try:
        stats = client.kb_stats()["collections"]
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    for item in stats:
        print(f"集合 {item['collection']}")
        print(f"  块数　　　: {item['chunks']}")
        print(f"  切分参数　: chunk_size={item['chunk_size']}  chunk_overlap={item['chunk_overlap']}")
        print(f"  向量维度　: {item['embed_dim']}")
        if item.get("bm25"):
            print(f"  BM25 索引 : {item['bm25']}")
        if item.get("avg_chunk_chars"):
            print(f"  平均块长　: {item['avg_chunk_chars']} 字")
        print(f"  最近更新　: {item.get('updated_at')}")

    answer = input("\n是否用 data/seed 重新建库？(y/N): ").strip().lower()
    if answer == "y":
        print("正在重建知识库（递归切分 + 重叠 + 向量化 + BM25）...")
        try:
            result = client.kb_rebuild()
        except BackendError as exc:
            print(f"❌ {exc}")
            return
        for item in result["results"]:
            print(f"  ✅ {item['collection']}：{item['documents']} 篇 → {item['chunks']} 块"
                  f"（{item['seconds']}s）")


def feature_context(client: BackendClient) -> None:
    print(f"\n{LINE}\n          会话长上下文压缩情况\n{LINE}")
    try:
        sessions = client.list_sessions()
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    if not sessions:
        print("暂无会话。可以先跑一次「模拟面试」。")
        return

    for index, session in enumerate(sessions, start=1):
        print(f"  {index}. [{session['mode']}] {session['title']} "
              f"消息 {session['message_count']} 条，"
              f"摘要 {session['summary_chars']} 字（upto seq={session['summary_upto_seq']}）")

    raw = input("\n请输入要查看的编号（回车返回）：").strip()
    if not raw.isdigit() or not 1 <= int(raw) <= len(sessions):
        return

    session_id = sessions[int(raw) - 1]["id"]
    try:
        context = client.session_context(session_id)
    except BackendError as exc:
        print(f"❌ {exc}")
        return

    stats = context.get("stats", {})
    print(f"\n🧠 {format_memory(stats)}")
    print(f"\n【滚动摘要】\n{context.get('summary') or '（尚未生成摘要）'}")
    print("\n【送给模型的上下文结构】")
    for message in context.get("messages", []):
        print(f"  - {message['role']:9} {message['chars']:>6} 字 :: "
              f"{message['preview'][:60]}...")


# ---------------------------------------------------------------------- #
# 主循环
# ---------------------------------------------------------------------- #
def show_menu(role_type: str, backend: str) -> None:
    print(MENU.format(line=LINE, role=ROLE_LABELS[role_type], backend=backend))


def main() -> int:
    client = BackendClient()
    print("正在连接后端...")
    try:
        health = client.health()
    except BackendError as exc:
        print(f"❌ {exc}")
        print("\n请先启动后端：python server.py")
        return 1

    print(f"✅ 已连接：{health['database']['backend']} 数据库"
          f"{'（SQLite 回退）' if health['database']['fallback'] else ''}，"
          f"知识库 {sum(item['chunks'] for item in health['knowledge_base'])} 块")
    if health["llm"]["mock_mode"]:
        print("⚠️  未配置 API Key，后端运行在离线 Mock 模式（返回桩数据）")
        print("    → 直接用菜单 10「模型设置」在线填入 Key，填完立即生效，不用改 .env、不用重启")

    role_type = choose_role_type()
    state: dict = {"target_job": None}

    try:
        while True:
            show_menu(role_type, client.base_url)
            choice = input("请选择功能（0-11）：").strip()

            if choice == "1":
                feature_resume(client, role_type, "optimize", "AI 简历优化（RAG 增强）", state)
            elif choice == "2":
                feature_resume(client, role_type, "questions", "AI 面试题生成（RAG 增强）", state)
            elif choice == "3":
                feature_mock_interview(client, role_type, state)
            elif choice == "4":
                feature_resume(client, role_type, "score", "AI 简历评分（RAG 增强）", state)
            elif choice == "5":
                feature_rag_chat(client, role_type)
            elif choice == "6":
                feature_tools(client, role_type, state)
            elif choice == "7":
                feature_search_and_pick_job(client, state)
            elif choice == "8":
                feature_knowledge(client)
            elif choice == "9":
                feature_context(client)
            elif choice == "10":
                feature_llm_settings(client)
            elif choice == "11":
                role_type = choose_role_type()
                continue
            elif choice == "0":
                print("\n再见！祝你求职顺利！")
                return 0
            else:
                print("\n无效选择，请输入 0-11 之间的数字")

            try:
                input("\n按回车键返回菜单...")
            except EOFError:
                return 0
    except KeyboardInterrupt:
        print("\n\n已中断，再见！")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
