"""用户求职档案与「必填门禁」。

这是整个单机版应用的起点。设计原则（按需求）：

    **求职方向 / 期望城市 / 期望薪资没填全，就不允许进入下一步。**

门禁分三级，前端据此做向导，后端每个接口再校验一次（不能只靠前端）：

    第 1 级 profile_ready      : 档案必填项齐全 + 已上传简历
    第 2 级 market_ready       : 已按档案抓过真实岗位（有市场数据）
    第 3 级 personal_kb_ready  : 已建好用户专属知识库
    另外 llm_ready             : 大模型已配置（否则什么都跑不了）

``readiness()`` 返回一个结构化状态，既给前端渲染向导，
也给出「下一步该做什么」和「缺什么」的明确文案。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from .db import repository, session_scope

logger = logging.getLogger("ai_resume_helper.profile")

PROFILE_ID = 1  # 单机版只有一行

# 必填项：字段名 -> 给用户看的名字
REQUIRED_FIELDS: tuple[tuple[str, str], ...] = (
    ("target_role", "求职方向"),
    ("expect_city", "期望城市"),
    ("expect_salary_min", "期望薪资下限"),
)
RESUME_FIELD = ("has_resume", "简历")


@dataclass
class UserProfileData:
    """档案的内存表示。"""

    target_role: str = ""
    expect_city: str = ""
    expect_salary_min: float = 0.0
    expect_salary_max: float = 0.0
    resume_id: str | None = None
    experience_years: float = 0.0
    education: str = ""
    skills: list[str] = field(default_factory=list)
    target_keywords: list[str] = field(default_factory=list)
    extra_notes: str = ""
    market_summary: dict[str, Any] = field(default_factory=dict)
    market_updated_at: str | None = None
    target_job: dict[str, Any] = field(default_factory=dict)
    updated_at: str | None = None

    @property
    def has_resume(self) -> bool:
        return bool(self.resume_id)

    @property
    def salary_text(self) -> str:
        low, high = self.expect_salary_min, self.expect_salary_max
        if low and high:
            return f"{low:g}-{high:g}K"
        if low:
            return f"{low:g}K 以上"
        if high:
            return f"{high:g}K 以下"
        return "未填写"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_resume"] = self.has_resume
        data["salary_text"] = self.salary_text
        return data

    def search_keywords(self) -> list[str]:
        """用于岗位检索的关键字：优先用户自己填的，否则从求职方向拆。"""
        if self.target_keywords:
            return list(self.target_keywords)
        return derive_keywords(self.target_role, self.skills)


# ---------------------------------------------------------------------- #
# 关键字推导
# ---------------------------------------------------------------------- #
# 从求职方向里剥掉这些「噪声词」，剩下的才是职位名里真正会出现的词
_NOISE_WORDS = (
    # 中文职级 / 岗位后缀
    "工程师", "开发工程师", "开发", "分析师", "经理", "专员", "主管", "总监",
    "助理", "专家", "岗位", "职位", "方向", "相关", "工作", "管理",
    "高级", "资深", "初级", "中级", "实习", "校招", "社招", "全职", "兼职",
    # 英文
    "engineer", "developer", "analyst", "manager", "specialist", "senior",
    "junior", "lead", "staff", "intern", "fulltime",
)


def derive_keywords(target_role: str, skills: list[str] | None = None) -> list[str]:
    """把「AI Agent 开发工程师」这类方向拆成可用于职位名匹配的关键字。

    返回结果按「信息量」排序，且包含原始方向本身 —— 有些官网的职位名
    就是完整的方向名，直接用原文匹配命中率最高。
    """
    role = (target_role or "").strip()
    keywords: list[str] = []

    if role:
        keywords.append(role.lower())

    # 拆词：中英混合统一按非字母数字汉字切
    import re

    parts = [p for p in re.split(r"[\s/,、|·\-—]+", role) if p]
    for part in parts:
        token = part.strip().lower()
        if not token or len(token) < 2:
            continue
        if token in _NOISE_WORDS:
            continue
        keywords.append(token)
        # 去掉「工程师/开发」这类后缀后再作为一轮候选
        stripped = token
        for noise in _NOISE_WORDS:
            if stripped.endswith(noise) and len(stripped) > len(noise) + 1:
                stripped = stripped[: -len(noise)]
                break
        if stripped and stripped != token and len(stripped) >= 2:
            keywords.append(stripped)

    for skill in skills or []:
        token = str(skill).strip().lower()
        if len(token) >= 2 and token not in keywords:
            keywords.append(token)

    # 去重保序
    seen: list[str] = []
    for item in keywords:
        if item not in seen:
            seen.append(item)
    return seen[:12]


# ---------------------------------------------------------------------- #
# 读写
# ---------------------------------------------------------------------- #
def _row_to_data(row: Any) -> UserProfileData:
    def loads(raw: str | None, default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default

    return UserProfileData(
        target_role=row.target_role or "",
        expect_city=row.expect_city or "",
        expect_salary_min=float(row.expect_salary_min or 0),
        expect_salary_max=float(row.expect_salary_max or 0),
        resume_id=row.resume_id,
        experience_years=float(row.experience_years or 0),
        education=row.education or "",
        skills=loads(row.skills_json, []),
        target_keywords=loads(row.target_keywords_json, []),
        extra_notes=row.extra_notes or "",
        market_summary=loads(row.market_summary_json, {}),
        market_updated_at=(
            row.market_updated_at.strftime("%Y-%m-%d %H:%M:%S")
            if row.market_updated_at
            else None
        ),
        target_job=loads(row.target_job_json, {}),
        updated_at=row.updated_at.strftime("%Y-%m-%d %H:%M:%S") if row.updated_at else None,
    )


async def load_profile(db: AsyncSession | None = None) -> UserProfileData:
    """读取档案；不存在时返回空档案（不抛异常，方便前端首次渲染）。"""
    if db is not None:
        row = await repository.get_user_profile(db, PROFILE_ID)
        return _row_to_data(row) if row else UserProfileData()
    async with session_scope() as session:
        row = await repository.get_user_profile(session, PROFILE_ID)
        return _row_to_data(row) if row else UserProfileData()


async def save_profile(payload: dict[str, Any], db: AsyncSession | None = None) -> UserProfileData:
    """保存档案（部分更新：只覆盖传进来的字段）。"""
    fields = {
        "target_role",
        "expect_city",
        "expect_salary_min",
        "expect_salary_max",
        "resume_id",
        "experience_years",
        "education",
        "extra_notes",
    }
    json_fields = {"skills": "skills_json", "target_keywords": "target_keywords_json"}

    values: dict[str, Any] = {}
    for key in fields:
        if key in payload and payload[key] is not None:
            values[key] = payload[key]
    for key, column in json_fields.items():
        if key in payload and payload[key] is not None:
            values[column] = json.dumps(payload[key], ensure_ascii=False)

    if "target_role" in values and "target_keywords_json" not in values:
        # 方向变了，关键字跟着重算（除非用户显式传了 keywords）
        values["target_keywords_json"] = json.dumps(
            derive_keywords(str(values["target_role"])), ensure_ascii=False
        )

    if db is not None:
        row = await repository.upsert_user_profile(db, PROFILE_ID, values)
        return _row_to_data(row)
    async with session_scope() as session:
        row = await repository.upsert_user_profile(session, PROFILE_ID, values)
        return _row_to_data(row)


async def save_market_result(
    summary: dict[str, Any],
    target_job: dict[str, Any] | None = None,
    db: AsyncSession | None = None,
) -> None:
    values: dict[str, Any] = {
        "market_summary_json": json.dumps(summary, ensure_ascii=False),
        "market_updated_at": datetime.now(),
    }
    if target_job is not None:
        values["target_job_json"] = json.dumps(target_job, ensure_ascii=False)
    if db is not None:
        await repository.upsert_user_profile(db, PROFILE_ID, values)
        return
    async with session_scope() as session:
        await repository.upsert_user_profile(session, PROFILE_ID, values)


# ---------------------------------------------------------------------- #
# 门禁
# ---------------------------------------------------------------------- #
def missing_required(profile: UserProfileData) -> list[str]:
    """返回还没填的必填项（给用户看的中文名）。"""
    missing: list[str] = []
    for name, label in REQUIRED_FIELDS:
        value = getattr(profile, name, None)
        if name.startswith("expect_salary"):
            if not value or float(value) <= 0:
                missing.append(label)
        elif not str(value or "").strip():
            missing.append(label)
    if not profile.has_resume:
        missing.append(RESUME_FIELD[1])
    return missing


async def readiness(profile: UserProfileData | None = None) -> dict[str, Any]:
    """综合判定当前进度，驱动前端向导与后端门禁。"""
    from .llm_config import get_llm_config
    from rag import get_knowledge_base

    profile = profile or await load_profile()
    missing = missing_required(profile)
    profile_ready = not missing

    llm = get_llm_config()
    llm_ready = llm.configured

    market_ready = bool(profile.market_summary.get("sample_size"))

    personal_chunks = 0
    try:
        kb = get_knowledge_base()
        personal_chunks = await kb.acount(getattr(kb.cfg, "rag_collection_personal", "personal_kb"))
    except Exception as exc:  # noqa: BLE001 - 知识库尚未初始化时不影响档案判定
        logger.debug("读取个人知识库块数失败：%s", exc)
    personal_kb_ready = personal_chunks > 0

    blockers: list[str] = []
    if not llm_ready:
        blockers.append("还没有配置大模型 API Key（右上角「模型设置」）")
    if missing:
        blockers.append("档案还没填全：" + "、".join(missing))
    if not market_ready:
        blockers.append("还没有抓取过岗位市场数据")
    if not personal_kb_ready:
        blockers.append("还没有生成你的专属知识库")

    if not llm_ready:
        next_step = "llm"
    elif missing:
        next_step = "profile"
    elif not market_ready:
        next_step = "market"
    elif not personal_kb_ready:
        next_step = "knowledge_base"
    else:
        next_step = "ready"

    return {
        "profile": profile.to_dict(),
        "llm_ready": llm_ready,
        "profile_ready": profile_ready,
        "market_ready": market_ready,
        "personal_kb_ready": personal_kb_ready,
        "personal_kb_chunks": personal_chunks,
        "missing_fields": missing,
        "blockers": blockers,
        "next_step": next_step,
        "can_run_interview": bool(llm_ready and profile_ready and market_ready and personal_kb_ready),
        "can_crawl_market": bool(llm_ready and profile_ready),
    }


class ProfileIncompleteError(RuntimeError):
    """档案不完整时抛出，由 API 层转成 428。"""

    def __init__(self, missing: list[str], message: str = "") -> None:
        self.missing = missing
        super().__init__(message or ("档案信息不完整，缺少：" + "、".join(missing)))


async def require_profile_ready(profile: UserProfileData | None = None) -> UserProfileData:
    """门禁：档案必填项不齐就抛异常。抓取市场前调用。"""
    profile = profile or await load_profile()
    missing = missing_required(profile)
    if missing:
        raise ProfileIncompleteError(missing)
    return profile


async def require_interview_ready(profile: UserProfileData | None = None) -> dict[str, Any]:
    """门禁：模拟面试/出题/简历优化前调用，要求档案+市场+个人知识库都就绪。"""
    state = await readiness(profile)
    if not state["can_run_interview"]:
        raise ProfileIncompleteError(state["missing_fields"], "；".join(state["blockers"]))
    return state


__all__ = [
    "PROFILE_ID",
    "ProfileIncompleteError",
    "UserProfileData",
    "derive_keywords",
    "load_profile",
    "missing_required",
    "readiness",
    "require_interview_ready",
    "require_profile_ready",
    "save_market_result",
    "save_profile",
]
