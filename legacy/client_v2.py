"""后端 HTTP 客户端（命令行版与网页版共用）。

只依赖 ``httpx``，同步 API，方便在 CLI / Streamlit 这类同步环境里使用。
所有方法在非 2xx 时抛 ``BackendError``，并把后端的 ``detail`` 原样带出来，
方便界面直接展示。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import httpx

from config import get_settings


class BackendError(RuntimeError):
    """后端返回错误或不可达。"""


class BackendClient:
    """AI 面试与简历助手后端客户端。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.backend_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.service_api_key
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BackendClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise BackendError(
                f"无法连接后端 {self.base_url}。请先启动后端：python server.py"
            ) from exc
        except httpx.TimeoutException as exc:
            raise BackendError(f"请求超时：{method} {path}") from exc

        if response.status_code >= 400:
            detail: Any
            try:
                detail = response.json().get("detail", response.text)
            except (json.JSONDecodeError, ValueError):
                detail = response.text
            if isinstance(detail, (dict, list)):
                detail = json.dumps(detail, ensure_ascii=False)
            raise BackendError(f"HTTP {response.status_code}：{detail}")
        if not response.content:
            return None
        return response.json()

    # ================= 基础 ================= #
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics")

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/stats")

    # ================= 知识库 ================= #
    def kb_stats(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/knowledge/stats")

    def kb_search(
        self,
        query: str,
        collections: list[str] | None = None,
        top_k: int = 5,
        role_type: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if collections:
            payload["collections"] = collections
        if role_type:
            payload["role_type"] = role_type
        return self._request("POST", "/api/v1/knowledge/search", json=payload)

    def kb_rebuild(self, collection: str | None = None) -> dict[str, Any]:
        return self._request(
            "POST", "/api/v1/knowledge/rebuild", json={"collection": collection, "from_seed": True}
        )

    def kb_add(self, collection: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/knowledge/documents",
            json={"collection": collection, "documents": documents},
        )

    # ================= RAG / 工具 ================= #
    def rag_ask(
        self,
        question: str,
        role_type: str = "tech",
        top_k: int | None = None,
        collections: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question, "role_type": role_type}
        if top_k:
            payload["top_k"] = top_k
        if collections:
            payload["collections"] = collections
        return self._request("POST", "/api/v1/rag/ask", json=payload)

    def tools_ask(
        self, question: str, role_type: str = "tech", use_tools: bool = True, strict: bool = False
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/tools/ask",
            json={
                "question": question,
                "role_type": role_type,
                "use_tools": use_tools,
                "strict": strict,
            },
        )

    def list_tools(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/tools")["tools"]

    def salary(
        self,
        role: str,
        city: str | None = None,
        level: str | None = None,
        years: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": role}
        if city:
            payload["city"] = city
        if level:
            payload["level"] = level
        if years is not None:
            payload["years"] = years
        return self._request("POST", "/api/v1/tools/salary", json=payload)

    def jobs(
        self,
        keywords: str | None = None,
        city: str | None = None,
        sources: str | None = None,
        limit: int = 8,
        with_description: bool = True,
    ) -> dict[str, Any]:
        """按关键字检索真实在招岗位（关键字只匹配职位名称）。"""
        payload: dict[str, Any] = {
            "keywords": keywords or "",
            "limit": limit,
            "with_description": with_description,
        }
        if city:
            payload["city"] = city
        if sources:
            payload["sources"] = sources
        return self._request("POST", "/api/v1/tools/jobs", json=payload)

    def job_sources(self) -> dict[str, Any]:
        """岗位数据源清单（说明数据到底来自哪里）。"""
        return self._request("GET", "/api/v1/tools/job-sources")

    # ================= 运行时设置（在线填 API Key） ================= #
    def llm_settings(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/settings/llm")

    def update_llm_settings(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        persist: bool = True,
        verify: bool = True,
    ) -> dict[str, Any]:
        """在线修改大模型配置，立即生效（可写回 .env 并当场验证）。"""
        payload: dict[str, Any] = {"persist": persist, "verify": verify}
        if api_key is not None:
            payload["api_key"] = api_key
        if base_url:
            payload["base_url"] = base_url
        if model:
            payload["model"] = model
        if temperature is not None:
            payload["temperature"] = temperature
        return self._request("PUT", "/api/v1/settings/llm", json=payload)

    def verify_llm(self) -> dict[str, Any]:
        return self._request("POST", "/api/v1/settings/llm/verify")

    def clear_llm_key(self, persist: bool = True) -> dict[str, Any]:
        return self._request(
            "DELETE", "/api/v1/settings/llm/key", params={"persist": str(persist).lower()}
        )

    # ================= 简历 ================= #
    def create_resume(self, filename: str, content: str) -> dict[str, Any]:
        return self._request(
            "POST", "/api/v1/resumes", json={"filename": filename, "content": content}
        )

    def upload_resume(self, pdf_path: str | Path) -> dict[str, Any]:
        path = Path(pdf_path)
        with path.open("rb") as handle:
            files = {"file": (path.name, handle, "application/pdf")}
            return self._request("POST", "/api/v1/resumes/upload", files=files)

    def upload_resume_file(self, filename: str, data: bytes) -> dict[str, Any]:
        """上传内存中的 PDF 字节（Streamlit 上传的文件对象用这个）。"""
        files = {"file": (filename or "resume.pdf", data, "application/pdf")}
        return self._request("POST", "/api/v1/resumes/upload", files=files)

    def resume_feature(
        self,
        feature: str,
        role_type: str = "tech",
        resume_id: str | None = None,
        resume_text: str | None = None,
        job_context: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"role_type": role_type}
        if resume_id:
            payload["resume_id"] = resume_id
        if resume_text:
            payload["resume_text"] = resume_text
        if job_context:
            payload["job_context"] = job_context
        return self._request("POST", f"/api/v1/resume/{feature}", json=payload)

    def optimize_resume(self, **kw: Any) -> dict[str, Any]:
        return self.resume_feature("optimize", **kw)

    def interview_questions(self, **kw: Any) -> dict[str, Any]:
        return self.resume_feature("questions", **kw)

    def score_resume(self, **kw: Any) -> dict[str, Any]:
        return self.resume_feature("score", **kw)

    # ================= 多轮对话 ================= #
    def create_session(
        self,
        mode: str = "mock_interview",
        role_type: str = "tech",
        title: str = "",
        resume_id: str | None = None,
        resume_text: str | None = None,
        job_title: str = "",
        job_company: str = "",
        job_context: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": mode, "role_type": role_type, "title": title}
        if resume_id:
            payload["resume_id"] = resume_id
        if resume_text:
            payload["resume_text"] = resume_text
        if job_title:
            payload["job_title"] = job_title
        if job_company:
            payload["job_company"] = job_company
        if job_context:
            payload["job_context"] = job_context
        return self._request("POST", "/api/v1/sessions", json=payload)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/sessions", params={"limit": limit})["sessions"]

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/sessions/{session_id}")

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/api/v1/sessions/{session_id}/messages")["messages"]

    def start_session(self, session_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/sessions/{session_id}/start")

    def send_message(
        self, session_id: str, content: str, use_tools: bool = False
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/messages",
            json={"content": content, "use_tools": use_tools},
        )

    def session_context(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/sessions/{session_id}/context")

    def session_report(self, session_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/sessions/{session_id}/report")

    def delete_session(self, session_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/sessions/{session_id}")


# ---------------------------------------------------------------------- #
# 展示辅助
# ---------------------------------------------------------------------- #
def format_sources(sources: Iterable[dict[str, Any]], preview: int = 160) -> str:
    """把引用来源渲染成终端可读文本。"""
    lines: list[str] = []
    for source in sources:
        similarity = source.get("similarity")
        similarity_text = f"相似度 {similarity:.3f}" if isinstance(similarity, float) else "关键词命中"
        lines.append(f"  [{source.get('index')}] {source.get('label')}（{similarity_text}）")
        text = " ".join(str(source.get("text", "")).split())
        if text:
            lines.append(f"      {text[:preview]}{'…' if len(text) > preview else ''}")
    return "\n".join(lines)


def format_target_job(job: dict[str, Any] | None) -> str:
    """把选定的目标岗位渲染成可放进 Prompt 的 JD 文本。"""
    if not job:
        return ""
    lines = [f"{job.get('title', '')}" + (f" @ {job['company']}" if job.get("company") else "")]
    if job.get("location"):
        lines.append(f"工作地点：{job['location']}")
    if job.get("salary"):
        lines.append(f"薪资：{job['salary']}")
    if job.get("job_type"):
        lines.append(f"类型：{job['job_type']}")
    if job.get("tags"):
        lines.append("标签/技能：" + "、".join(str(t) for t in job["tags"]))
    if job.get("description"):
        lines.append("岗位描述：" + str(job["description"]))
    if job.get("url"):
        lines.append(f"原文链接：{job['url']}")
    lines.append(f"数据来源：{job.get('source', '未知')}")
    return "\n".join(lines)


def format_tool_calls(tool_calls: Iterable[dict[str, Any]]) -> str:
    """把工具调用链渲染成终端可读文本。"""
    lines: list[str] = []
    for index, call in enumerate(tool_calls, start=1):
        status = "✅" if call.get("ok") else "❌"
        lines.append(
            f"  {status} {index}. {call.get('name')}({json.dumps(call.get('arguments'), ensure_ascii=False)})"
            f"  {call.get('elapsed_ms')}ms"
        )
        if not call.get("ok"):
            lines.append(f"      错误：{call.get('error')}")
        else:
            result = call.get("result") or {}
            if isinstance(result, dict):
                source = result.get("source")
                if source:
                    lines.append(f"      数据来源：{source}")
    return "\n".join(lines)


def format_confidence(result: dict[str, Any]) -> str:
    """渲染防幻觉指标。"""
    confidence = result.get("confidence", 0.0)
    report = result.get("citation_report") or {}
    parts = [f"可信度 {confidence:.0%}"]
    if report:
        parts.append(f"引用覆盖 {report.get('citation_rate', 0):.0%}")
        if report.get("invalid"):
            parts.append(f"无效引用 {report['invalid']}")
    retrieval = result.get("retrieval") or {}
    if retrieval.get("hits") is not None:
        parts.append(f"检索命中 {retrieval['hits']} 条 / {retrieval.get('elapsed_ms', 0)}ms")
    return " | ".join(parts)


def format_memory(memory: dict[str, Any] | None) -> str:
    """渲染长上下文压缩指标。"""
    if not memory:
        return "（无长上下文统计）"
    return (
        f"历史 {memory.get('total_messages')} 条 / {memory.get('history_chars')} 字 → "
        f"摘要 {memory.get('summary_chars')} 字（覆盖 {memory.get('summarized_messages')} 条）"
        f" + 窗口原文 {memory.get('window_messages')} 条 / {memory.get('window_chars')} 字"
        f"，压缩比 {memory.get('compressed_ratio', 0):.0%}"
    )
