"""文档加载器：把 PDF / Markdown / TXT / JSONL 统一变成 Document。

种子数据（``data/seed/*.jsonl``）会被渲染成便于检索的富文本段落，
让每个块都是「自解释」的：单独看一块也能知道它讲的是哪个岗位、
哪类问题，从而提升检索命中率。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pdf_reader import extract_text_from_pdf, is_valid_resume_text


@dataclass(slots=True)
class Document:
    """一篇待入库的文档。"""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# 通用文件加载
# ---------------------------------------------------------------------- #
def load_text_file(path: str | Path, metadata: dict[str, Any] | None = None) -> Document:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = {"source": path.name, "source_path": str(path)}
    meta.update(metadata or {})
    return Document(text=text, metadata=meta)


def load_pdf(path: str | Path, metadata: dict[str, Any] | None = None) -> Document | None:
    path = Path(path)
    text = extract_text_from_pdf(str(path))
    if not is_valid_resume_text(text):
        return None
    meta = {"source": path.name, "source_path": str(path)}
    meta.update(metadata or {})
    return Document(text=text, metadata=meta)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL（每行一个 JSON 对象）。"""
    path = Path(path)
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise ValueError(f"{path.name} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    return records


def load_any(path: str | Path, metadata: dict[str, Any] | None = None) -> list[Document]:
    """按后缀自动选择加载方式。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = load_pdf(path, metadata)
        return [doc] if doc else []
    if suffix in {".md", ".txt", ".py", ".json"}:
        return [load_text_file(path, metadata)]
    if suffix in {".jsonl", ".ndjson"}:
        return [
            Document(
                text=json.dumps(record, ensure_ascii=False, indent=2),
                metadata={"source": path.name, **(metadata or {})},
            )
            for record in load_jsonl(path)
        ]
    raise ValueError(f"不支持的文件类型：{suffix}")


# ---------------------------------------------------------------------- #
# 种子数据渲染
# ---------------------------------------------------------------------- #
def render_interview_question(record: dict[str, Any]) -> Document:
    """把一条面试题记录渲染成富文本文档。"""
    lines: list[str] = []
    role = record.get("role", "")
    category = record.get("category", "")
    difficulty = record.get("difficulty", "")
    lines.append(f"【知识库】面试题库｜岗位：{role}｜类别：{category}｜难度：{difficulty}")

    if record.get("title"):
        lines.append(f"标题：{record['title']}")

    lines.append(f"问题：{record.get('question', '').strip()}")

    points = record.get("answer_points") or []
    if points:
        lines.append("参考答案要点：")
        for i, point in enumerate(points, 1):
            lines.append(f"  {i}. {point}")

    followups = record.get("followups") or []
    if followups:
        lines.append("常见追问：")
        for i, item in enumerate(followups, 1):
            lines.append(f"  {i}. {item}")

    tags = record.get("tags") or []
    if tags:
        lines.append("关键词：" + "、".join(str(t) for t in tags))

    metadata = {
        "doc_type": "interview_question",
        "record_id": str(record.get("id", "")),
        "title": str(record.get("title") or record.get("question", ""))[:80],
        "role_type": str(record.get("role_type", "")),
        "role": str(role),
        "category": str(category),
        "difficulty": str(difficulty),
        "tags": "、".join(str(t) for t in tags),
        "source": "面试题库",
    }
    return Document(text="\n".join(lines), metadata=metadata)


def render_job_description(record: dict[str, Any]) -> Document:
    """把一条岗位 JD 记录渲染成富文本文档。"""
    lines: list[str] = []
    title = record.get("title", "")
    company = record.get("company", "")
    city = record.get("city", "")
    lines.append(
        f"【知识库】岗位JD｜职位：{title}｜公司：{company}｜城市：{city}"
    )
    lines.append(
        f"薪资：{record.get('salary', '面议')}｜经验：{record.get('experience', '不限')}"
        f"｜学历：{record.get('education', '不限')}"
    )

    skills = record.get("skills") or []
    if skills:
        lines.append("技能要求：" + "、".join(str(s) for s in skills))

    responsibilities = record.get("responsibilities") or []
    if responsibilities:
        lines.append("岗位职责：")
        for i, item in enumerate(responsibilities, 1):
            lines.append(f"  {i}. {item}")

    requirements = record.get("requirements") or []
    if requirements:
        lines.append("任职要求：")
        for i, item in enumerate(requirements, 1):
            lines.append(f"  {i}. {item}")

    if record.get("description"):
        lines.append("补充说明：" + str(record["description"]))

    metadata = {
        "doc_type": "job_description",
        "record_id": str(record.get("id", "")),
        "title": str(title),
        "role_type": str(record.get("role_type", "")),
        "company": str(company),
        "city": str(city),
        "salary": str(record.get("salary", "")),
        "experience": str(record.get("experience", "")),
        "education": str(record.get("education", "")),
        "tags": "、".join(str(s) for s in skills),
        "source": "岗位JD库",
    }
    return Document(text="\n".join(lines), metadata=metadata)


def load_seed_documents(seed_dir: str | Path) -> dict[str, list[Document]]:
    """加载全部种子数据，返回 {集合名: [Document]}。"""
    from config import get_settings

    cfg = get_settings()
    seed_dir = Path(seed_dir)
    result: dict[str, list[Document]] = {
        cfg.rag_collection_questions: [],
        cfg.rag_collection_jobs: [],
    }

    questions_file = seed_dir / "interview_questions.jsonl"
    if questions_file.exists():
        result[cfg.rag_collection_questions] = [
            render_interview_question(record) for record in load_jsonl(questions_file)
        ]

    jobs_file = seed_dir / "job_descriptions.jsonl"
    if not jobs_file.exists():
        # 岗位库与 Function Calling 的岗位搜索工具共用同一份数据
        jobs_file = seed_dir / "job_postings.jsonl"
    if jobs_file.exists():
        result[cfg.rag_collection_jobs] = [
            render_job_description(record) for record in load_jsonl(jobs_file)
        ]

    return result


def iter_directory(path: str | Path, pattern: str = "*") -> Iterable[Path]:
    path = Path(path)
    return (p for p in sorted(path.glob(pattern)) if p.is_file())
