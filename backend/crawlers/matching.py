"""职位名 / 城市匹配。

需求明确：**关键字只匹配职位名称**，不做全字段模糊匹配。
所以搜公司名不该命中，搜「Agent」「python后端」才该命中。

多个关键字之间是「或」：命中任意一个就算。
中文关键字对英文站点无效，因此这里也负责中英同义词扩展。
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# 中文 -> 英文同义词（用于英文岗位源）
KEYWORD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "python后端": ("python backend", "backend engineer", "backend developer", "python engineer"),
    "后端": ("backend", "back-end", "back end"),
    "前端": ("frontend", "front-end", "react", "vue"),
    "全栈": ("full stack", "fullstack"),
    "算法": ("algorithm", "machine learning", "ml engineer", "data scientist"),
    "大模型": ("llm", "large language model", "generative ai", "genai"),
    "ai": ("ai", "artificial intelligence", "machine learning"),
    "人工智能": ("ai", "artificial intelligence", "machine learning"),
    "agent": ("agent", "agentic", "llm"),
    "智能体": ("agent", "agentic"),
    "数据分析": ("data analyst", "data analytics", "business intelligence"),
    "数据": ("data",),
    "测试": ("qa", "test", "sdet", "quality assurance"),
    "运维": ("devops", "sre", "site reliability", "platform engineer"),
    "产品经理": ("product manager", "product owner"),
    "运营": ("operations", "growth", "community"),
    "市场": ("marketing", "growth"),
    "销售": ("sales", "account executive"),
    "设计": ("designer", "design", "ux", "ui"),
    "安全": ("security", "appsec", "infosec"),
    "嵌入式": ("embedded", "firmware"),
    "移动端": ("mobile", "android", "ios"),
    "java": ("java",),
    "golang": ("golang", "go engineer"),
    "c++": ("c++", "cpp"),
    "rust": ("rust",),
    "react": ("react",),
    "kubernetes": ("kubernetes", "k8s"),
}

_SPLIT_RE = re.compile(r"[,，、|/\s]+")
_ASCII_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.]*")

# 常见城市（含常见写法），用于城市匹配与自动识别
CITY_ALIASES: dict[str, tuple[str, ...]] = {
    "北京": ("北京", "北京市", "beijing"),
    "上海": ("上海", "上海市", "shanghai"),
    "广州": ("广州", "广州市", "guangzhou"),
    "深圳": ("深圳", "深圳市", "shenzhen"),
    "杭州": ("杭州", "杭州市", "hangzhou"),
    "成都": ("成都", "成都市", "chengdu"),
    "武汉": ("武汉", "武汉市", "wuhan"),
    "南京": ("南京", "南京市", "nanjing"),
    "西安": ("西安", "西安市", "xian", "xi'an"),
    "苏州": ("苏州", "苏州市", "suzhou"),
    "天津": ("天津", "天津市", "tianjin"),
    "重庆": ("重庆", "重庆市", "chongqing"),
    "长沙": ("长沙", "长沙市", "changsha"),
    "郑州": ("郑州", "郑州市", "zhengzhou"),
    "青岛": ("青岛", "青岛市", "qingdao"),
    "合肥": ("合肥", "合肥市", "hefei"),
    "厦门": ("厦门", "厦门市", "xiamen"),
    "福州": ("福州", "福州市", "fuzhou"),
    "济南": ("济南", "济南市", "jinan"),
    "大连": ("大连", "大连市", "dalian"),
    "宁波": ("宁波", "宁波市", "ningbo"),
    "无锡": ("无锡", "无锡市", "wuxi"),
    "佛山": ("佛山", "佛山市", "foshan"),
    "东莞": ("东莞", "东莞市", "dongguan"),
    "珠海": ("珠海", "珠海市", "zhuhai"),
    "远程": ("远程", "remote", "anywhere"),
    "全国": ("全国", "多地", "不限", "multiple locations"),
}


def split_keywords(raw: str | Sequence[str] | None) -> list[str]:
    """把用户/模型给的关键字串拆成列表（逗号、顿号、空格、竖线都支持）。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = _SPLIT_RE.split(raw)
    else:
        parts = []
        for item in raw:
            parts.extend(_SPLIT_RE.split(str(item)))
    result: list[str] = []
    for part in parts:
        token = part.strip().lower()
        if token and token not in result:
            result.append(token)
    return result


def expand_keywords(keywords: Sequence[str]) -> tuple[list[str], dict[str, list[str]]]:
    """为中英混合语料扩展同义词。

    返回 ``(全部匹配词, {原关键字: 扩展结果})``。

    注意：入参可能来自用户手输（``AI Agent``、``Python后端``），
    必须**先统一小写**再做同义词查表，否则大写形式会匹配不到词表。
    """
    expanded: dict[str, list[str]] = {}
    pool: list[str] = []
    for raw_keyword in keywords:
        keyword = str(raw_keyword or "").strip().lower()
        if not keyword:
            continue
        variants: list[str] = [keyword]
        for cn, en_list in KEYWORD_SYNONYMS.items():
            if cn in keyword:
                variants.extend(en_list)
        # 「python后端开发」这类混合词，抽出纯英文部分（python 本身就能匹配）
        ascii_part = " ".join(_ASCII_WORD_RE.findall(keyword))
        if ascii_part:
            variants.append(ascii_part)

        deduped: list[str] = []
        for variant in variants:
            token = variant.strip().lower()
            if token and token not in deduped:
                deduped.append(token)
        expanded[keyword] = deduped
        for token in deduped:
            if token not in pool:
                pool.append(token)
    return pool, expanded


def search_terms(keywords: Sequence[str], limit: int = 4) -> list[str]:
    """把关键字整理成「拿去搜索引擎用的检索词」，按命中率从高到低排序。

    实测经验（牛客网）：
        ``Java后端开发``   只搜到 1 条
        ``Java``          搜到 225 条
        ``全栈开发工程师`` 只搜到 24 条
        ``全栈``          搜到 51 条

    也就是说**完整的方向名往往太长、搜不到东西**。所以这里按
    「完整方向 → 剥后缀的核心词 → 纯英文技术名词 → 英文同义词」排序，
    让适配器逐级放宽、合并结果。太泛的词（如只搜「开发」）排在最后，
    由本地职位名匹配再筛一遍。
    """
    ordered: list[str] = []
    for keyword in keywords:
        token = str(keyword or "").strip().lower()
        if token and token not in ordered:
            ordered.append(token)

    # 纯英文技术名词（java / python / ai 这类）通常搜得最广
    for keyword in keywords:
        for word in _ASCII_WORD_RE.findall(str(keyword or "").lower()):
            if len(word) >= 2 and word not in ordered:
                ordered.append(word)

    # 英文同义词（中文方向在英文数据源上的兜底）
    pool, _ = expand_keywords(keywords)
    for token in pool:
        if token not in ordered:
            ordered.append(token)

    # 丢掉纯噪声词（「开发」「工程师」这种搜出来全是无关岗位）
    noisy = {"开发", "工程师", "专员", "经理", "岗位", "职位", "engineer", "developer"}
    cleaned = [t for t in ordered if t not in noisy] or ordered
    return cleaned[:limit]


def title_matches(title: str, match_terms: Sequence[str]) -> list[str]:
    """返回职位名命中的关键字（空列表表示不匹配）。"""
    lowered = (title or "").lower()
    if not lowered:
        return []
    # 把职位名里的符号压掉，让 "python 开发" 也能匹配 "python开发"
    compact = re.sub(r"[\s\-_/·]+", "", lowered)
    hits: list[str] = []
    for term in match_terms:
        if not term:
            continue
        if term in lowered or re.sub(r"[\s\-_/·]+", "", term) in compact:
            hits.append(term)
    return hits


def city_matches(text: str, city: str) -> bool:
    """城市匹配：支持「北京」匹配「北京市-北京市」这类写法。"""
    if not city:
        return True
    haystack = (text or "").lower()
    if not haystack:
        return False
    target = city.strip().lower()
    candidates = {target}
    for canonical, aliases in CITY_ALIASES.items():
        if target in aliases or target == canonical:
            candidates.update(alias.lower() for alias in aliases)
            candidates.add(canonical.lower())
            break
    return any(item and item in haystack for item in candidates)


def normalize_city(raw: str, fallback: str = "") -> str:
    """把「北京市-北京市」这类写法归一到「北京」。"""
    if not raw:
        return fallback
    text = str(raw)
    for canonical, aliases in CITY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in text.lower():
                return canonical
    return text.strip()[:32] or fallback


def dedupe_by_title(jobs: Iterable, max_per_title: int = 8) -> list:
    """同一个职位名保留最多 N 条（不同公司/城市），避免结果被一家公司刷屏。"""
    counts: dict[str, int] = {}
    result = []
    for job in jobs:
        title = (getattr(job, "title", "") or "").strip().lower()
        if counts.get(title, 0) >= max_per_title:
            continue
        counts[title] = counts.get(title, 0) + 1
        result.append(job)
    return result


__all__ = [
    "CITY_ALIASES",
    "KEYWORD_SYNONYMS",
    "city_matches",
    "dedupe_by_title",
    "expand_keywords",
    "normalize_city",
    "search_terms",
    "split_keywords",
    "title_matches",
]
