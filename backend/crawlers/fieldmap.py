"""「JSON 字段映射」的公共实现。

``render_api``（浏览器拦截 XHR）和 ``public_api.JsonApiCrawler``（直接请求接口）
拿到的都是**结构化的 JSON 对象**，只是取数据的方式不同。把「JSON 对象 → JobItem」
这段映射逻辑放在这里共用，避免两份实现慢慢长歪。

配置里 ``field_map`` 的键统一为：

    title / company / city / url / jd_text / salary_text / published_at
    / job_type / tags
    / salary_min_yuan / salary_max_yuan     数字薪资（元/月），优先于 salary_text
    / published_at_ms                       毫秒时间戳
    / degree / experience / company_size / company_type    会并进 tags

值支持四种写法，覆盖各家接口的怪形状：

1. **普通路径**：``"jobName"``、``"resultbody.job.items.0.name"``
2. **路径列表**：``["requirement", "description"]``
   —— 多段文本拼成一个字段（网易的 JD 就分成两段）
3. **模板串**：``"https://hr.163.com/job-detail.html?id={id}"``
   —— ``{...}`` 里是**该 JSON 对象内的路径**，用来拼出接口没直接给的链接
   （网易列表接口只给 id，不给孩子详情 URL，``beeUrl`` 实测恒为 null）
4. **列表字段**：JSON 里本身就是数组（如网易的 ``workPlaceNameList``）会自动拼接
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable

#: 一并塞进 tags 的附加字段
TAG_FIELDS = ("degree", "experience", "company_size", "company_type")

_TEMPLATE_RE = re.compile(r"\{([^{}]+)\}")


def json_path_get(value: Any, path: str) -> Any:
    """按 ``a.b.c`` 取值，支持列表下标（``items.0.name``）。取不到返回 None。"""
    if not path:
        return value
    current = value
    for part in str(path).split("."):
        if not part:
            continue
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            if not part.lstrip("-").isdigit():
                return None
            index = int(part)
            if not -len(current) <= index < len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def ms_to_day(value: Any) -> str:
    """毫秒时间戳 → ``YYYY-MM-DD``。"""
    try:
        return datetime.fromtimestamp(int(value) / 1000).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _stringify(value: Any, join: str = " / ") -> str:
    """把取到的值转成字符串；JSON 数组按 ``join`` 拼接。"""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [_stringify(item, join) for item in value]
        return join.join(part for part in parts if part)
    if isinstance(value, dict):
        return ""
    return str(value).strip()


def _resolve_one(item: dict[str, Any], spec: Any, join: str = " / ") -> str:
    """解析单个字段规格（路径 / 模板串）。"""
    text = str(spec)
    if "{" in text and "}" in text:
        # 模板串：{...} 里是 JSON 路径
        return _TEMPLATE_RE.sub(
            lambda match: _stringify(json_path_get(item, match.group(1)), join), text
        ).strip()
    return _stringify(json_path_get(item, text), join)


def pick_field(
    item: dict[str, Any], field_map: dict[str, Any], field: str, *, join: str = " / "
) -> str:
    """按 ``field_map`` 取出一个字段的字符串值（支持路径 / 列表 / 模板）。"""
    spec = field_map.get(field)
    if spec is None:
        return ""
    if isinstance(spec, list):
        parts = [_resolve_one(item, entry, join) for entry in spec]
        return join.join(part for part in parts if part)
    return _resolve_one(item, spec, join)


def pick_raw_field(item: dict[str, Any], field_map: dict[str, Any], field: str) -> Any:
    """取出字段的**原始** JSON 值（数组保持数组）。

    ``tags`` 需要原始数组：先转成字符串的话，``["五险", "双休"]`` 会变成
    ``"五险 / 双休"`` 这样一个标签，再按逗号切也切不开，标签就废了。
    """
    spec = field_map.get(field)
    if spec is None:
        return None
    if isinstance(spec, list):
        return [_resolve_one(item, entry) for entry in spec]
    if isinstance(spec, str) and "{" in spec and "}" in spec:
        return _resolve_one(item, spec)
    return json_path_get(item, spec)


def as_positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def clean_url(url: str, *, strip_query: bool = False) -> str:
    """按需剥掉 URL 的查询串。

    有些站点（如 51job）的详情链接带一个**每次检索都不同**的 ``req=`` 校验哈希，
    留着它会让同一个岗位在不同关键词下算出不同的 ``job_key``，白白产生重复。

    ⚠️ 对网易这类「详情页靠 query 参数定位」的站点**不能**开这个开关，
    否则 ``job-detail.html?id=58384`` 会被削成 ``job-detail.html``。
    """
    if not strip_query or not url:
        return url
    return url.split("?", 1)[0].split("#", 1)[0]


def collect_tags(raw: Any, extra_scalars: list[Any]) -> list[str]:
    """把各种形状的标签字段归一成 ``list[str]``。"""
    tags: list[str] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str) and entry.strip():
                tags.append(entry.strip())
            elif isinstance(entry, dict):
                name = entry.get("jobTagName") or entry.get("name") or entry.get("label")
                if name:
                    tags.append(str(name))
    elif isinstance(raw, str) and raw.strip():
        tags = [part.strip() for part in raw.split(",") if part.strip()]

    for value in extra_scalars:
        text = str(value or "").strip()
        if text:
            tags.append(text)

    # 去重但保序
    seen: set[str] = set()
    unique: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def build_job_from_mapping(
    make_job: Callable[..., Any],
    item: dict[str, Any],
    field_map: dict[str, Any],
    *,
    url_strip_query: bool = False,
    jd_limit: int = 6000,
) -> Any | None:
    """把一个 JSON 对象映射成 ``JobItem``；缺标题则返回 None。"""

    def pick(field: str, *, join: str = " / ") -> str:
        return pick_field(item, field_map, field, join=join)

    title = pick("title")
    if not title:
        return None

    # 数字薪资字段比文案字段可靠（不用去猜「6-8.5千」这种写法），优先使用
    salary_text = pick("salary_text")
    low_yuan = as_positive_float(pick("salary_min_yuan"))
    high_yuan = as_positive_float(pick("salary_max_yuan"))
    if low_yuan and high_yuan:
        salary_text = f"{low_yuan:g}-{high_yuan:g}元/月"

    published = pick("published_at") or ms_to_day(pick("published_at_ms"))

    kwargs: dict[str, Any] = {
        "title": title,
        "city": pick("city"),
        "job_type": pick("job_type"),
        "url": clean_url(pick("url"), strip_query=url_strip_query),
        "jd_text": pick("jd_text", join="\n")[:jd_limit],
        "salary_text": salary_text,
        "published_at": published[:32],
        # tags 用原始值：JSON 数组要保持数组，否则 ["五险","双休"] 会被
        # 拼成一个字符串，再也切不开
        "tags": collect_tags(
            pick_raw_field(item, field_map, "tags"),
            [pick(field) for field in TAG_FIELDS],
        ),
    }
    # company 只在**真的取到**时才传：留空才能让 make_job 用配置里的默认公司名
    company = pick("company")
    if company:
        kwargs["company"] = company

    return make_job(**kwargs)


__all__ = [
    "TAG_FIELDS",
    "as_positive_float",
    "build_job_from_mapping",
    "clean_url",
    "collect_tags",
    "json_path_get",
    "ms_to_day",
    "pick_field",
    "pick_raw_field",
]
