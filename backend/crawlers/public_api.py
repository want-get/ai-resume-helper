"""公开 JSON 接口类爬虫。

这类站点把岗位数据放在可直接请求的公开接口上（通常是自家 SPA 调用的 XHR），
不需要浏览器渲染，最快也最稳。每家公司的接口结构不同，所以一家一个适配器。

已适配：
    * ``jd``      京东校招  https://campus.jd.com/
    * ``jd_social`` 京东社招 https://zhaopin.jd.com/（有薪资字段）
    * ``nowcoder`` 牛客网    https://www.nowcoder.com/（有结构化薪资字段）
    * ``json_api`` **可配置**的通用 JSON 接口适配器，公司官网用它
      （已验证腾讯 careers.tencent.com、网易 hr.163.com）

新增一家公司要做的事：
    1. 打开该公司的招聘页，F12 → Network → 找到返回岗位列表的 XHR
    2. 照着下面任何一个适配器写一个类，实现 ``fetch(budget)``
    3. 注册进 ``ADAPTERS``，再到数据目录的 ``sources.json`` 里加一条 ``kind: public_api``
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from typing import Any

from .base import BaseCrawler, CrawlBudget, JobItem, SourceResult, _decode_response
from .fieldmap import build_job_from_mapping, json_path_get

logger = logging.getLogger("ai_resume_helper.crawlers.public_api")


def _ms_to_day(value: Any) -> str:
    try:
        return datetime.fromtimestamp(int(value) / 1000).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "null"} else text


class JDCampusCrawler(BaseCrawler):
    """京东校招：campus.jd.com 的公开分页接口。

    两个招聘类型：
        * ``present`` 校园招聘正式岗
        * ``talent``  技术人才计划（TGT 等，偏研究/大模型方向）
    """

    kind = "public_api"
    PAGE_API = "https://campus.jd.com/api/wx/position/page"
    PAGE_SIZE = 50
    JD_RAW_LIMIT = 8000
    TYPES = ("present", "talent")

    def _search_terms(self) -> list[str]:
        terms = [str(t).strip() for t in (self.config.get("_search_terms") or []) if str(t).strip()]
        if terms:
            return terms
        single = str(self.config.get("_search_keyword") or "").strip()
        return [single] if single else [""]

    def _fetch_type(
        self, recruit_type: str, budget: CrawlBudget, position_name: str = ""
    ) -> list[JobItem]:
        jobs: list[JobItem] = []
        seen: set[str] = set()
        page = 0
        total_pages = 1
        max_pages_per_term = int(self.config.get("max_pages") or 3)

        while page < min(total_pages, max_pages_per_term):
            budget.check()
            payload = {
                "pageSize": self.PAGE_SIZE,
                "pageIndex": page,
                "parameter": {
                    # 实测：positionName 是**生效的**服务端过滤
                    # （空=全量 124 条，"会计"=1 条，"算法"=12 条）
                    "positionName": position_name,
                    "planIdList": [],
                    "jobDirectionCodeList": [],
                    "workCityCodeList": [],
                    "positionDeptList": [],
                },
            }
            try:
                data = self.http_post_json(
                    f"{self.PAGE_API}?type={recruit_type}",
                    payload,
                    headers={"Referer": self.careers_url},
                    timeout=25,
                )
            except Exception as exc:  # noqa: BLE001 - 单页失败不放弃整体
                logger.warning("[%s] %s 第 %s 页失败：%s", self.label, recruit_type, page, exc)
                break

            body = (data or {}).get("body") or {}
            items = body.get("items") or []
            total = body.get("totalNumber") or len(items)
            try:
                total_pages = max(1, math.ceil(int(total) / self.PAGE_SIZE))
            except (TypeError, ValueError):
                total_pages = 1

            for item in items:
                title = _clean(item.get("positionName"))
                if not title:
                    continue
                publish_id = item.get("publishId") or item.get("reqId") or title
                unique = f"{recruit_type}:{publish_id}"
                if unique in seen:
                    continue
                seen.add(unique)

                cities: list[str] = []
                for requirement in item.get("requirementVoList") or []:
                    city = _clean(requirement.get("workCity"))
                    if city and city not in cities:
                        cities.append(city)

                duties = _clean(item.get("workContent"))
                qualification = _clean(item.get("qualification"))
                jd_text = "\n".join(
                    part
                    for part in (
                        "【岗位职责】" + duties if duties else "",
                        "【任职要求】" + qualification if qualification else "",
                    )
                    if part
                )

                jobs.append(
                    self.make_job(
                        title=title,
                        city=" / ".join(cities)[:80],
                        job_type="校招" if recruit_type == "present" else "人才计划",
                        # 接口只用于抓取；这个 SPA 路由是用户能点开的详情页
                        url=f"https://campus.jd.com/#/details?type={recruit_type}&id={publish_id}",
                        jd_text=jd_text[: self.JD_RAW_LIMIT],
                        published_at=_ms_to_day(item.get("publishTime")),
                        # 校招不公开薪资，留空由上层如实处理
                        salary_text="",
                        tags=[
                            tag
                            for tag in (
                                _clean(item.get("jobDirection")),
                                _clean(item.get("education")),
                                _clean(item.get("jobCategory")),
                            )
                            if tag
                        ],
                    )
                )

            if not items:
                break
            page += 1

        return jobs

    def fetch(self, budget: CrawlBudget) -> list[JobItem]:
        """按每个检索词分别过滤，再合并去重。

        京东的 positionName 过滤比较严格（"会计" 只返回 1 条），
        所以要多试几个词才不至于漏掉。
        """
        jobs: list[JobItem] = []
        seen: set[str] = set()
        for term in self._search_terms():
            for recruit_type in self.TYPES:
                if budget.remaining() <= 0:
                    break
                for job in self._fetch_type(recruit_type, budget, position_name=term):
                    key = job.job_key
                    if key in seen:
                        continue
                    seen.add(key)
                    jobs.append(job)

        logger.info("[%s] 抓到 %d 个岗位", self.label, len(jobs))
        return jobs


class JDSocialCrawler(BaseCrawler):
    """京东社招：zhaopin.jd.com 的公开岗位接口（含薪资区间）。"""

    kind = "public_api"
    LIST_API = "https://zhaopin.jd.com/api/position/list"
    PAGE_SIZE = 50

    def fetch(self, budget: CrawlBudget) -> list[JobItem]:
        jobs: list[JobItem] = []
        seen: set[str] = set()
        page = 1
        max_pages = int(self.config.get("max_pages") or 8)

        while page <= max_pages:
            budget.check()
            payload = {"pageNo": page, "pageSize": self.PAGE_SIZE, "positionName": ""}
            try:
                data = self.http_post_json(
                    self.LIST_API, payload, headers={"Referer": self.careers_url}, timeout=25
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] 第 %s 页失败：%s", self.label, page, exc)
                break

            body = data.get("body") if isinstance(data, dict) else None
            body = body if isinstance(body, dict) else (data or {})
            items = body.get("items") or body.get("list") or body.get("datas") or []
            if not items:
                break

            for item in items:
                title = _clean(item.get("positionName") or item.get("name"))
                if not title:
                    continue
                position_id = item.get("positionId") or item.get("id") or title
                unique = str(position_id)
                if unique in seen:
                    continue
                seen.add(unique)

                salary = _clean(
                    item.get("salary") or item.get("salaryDesc") or item.get("salaryRange")
                )
                duties = _clean(item.get("workContent") or item.get("positionDesc"))
                qualification = _clean(item.get("qualification") or item.get("requirement"))
                jd_text = "\n".join(
                    part
                    for part in (
                        "【岗位职责】" + duties if duties else "",
                        "【任职要求】" + qualification if qualification else "",
                    )
                    if part
                )

                jobs.append(
                    self.make_job(
                        title=title,
                        city=_clean(
                            item.get("workCity") or item.get("city") or item.get("workCityName")
                        ),
                        job_type="社招",
                        url=f"https://zhaopin.jd.com/web/job/detail?id={position_id}",
                        jd_text=jd_text[:8000],
                        salary_text=salary,
                        published_at=_ms_to_day(item.get("publishTime") or item.get("createTime")),
                        tags=[
                            tag
                            for tag in (
                                _clean(item.get("positionType")),
                                _clean(item.get("education")),
                                _clean(item.get("workYears")),
                            )
                            if tag
                        ],
                    )
                )

            if len(items) < self.PAGE_SIZE:
                break
            page += 1

        logger.info("[%s] 抓到 %d 个岗位", self.label, len(jobs))
        return jobs


class NowcoderCrawler(BaseCrawler):
    """牛客网职位广场：``np-api/u/job/square-search``。

    这是本项目**最重要**的数据源，因为只有它同时提供：

    * 真实中文岗位（校招 + 社招，覆盖各大公司、各行各业）
    * **结构化薪资字段**（``salaryMin`` / ``salaryMax`` / ``salaryMonth``），
      让「该城市该岗位的平均薪资」可以真的算出来，而不是猜的
    * 岗位职责 / 任职要求 / 技能关键词

    请求体：``{"recruitType": 1, "page": 1, "pageSize": 50, "query": "会计"}``

    ⚠️ 检索参数名是 **``query``**，不是 ``keyword``。
    用错了不会有任何报错，接口会**静默忽略**这个参数并返回默认的前 200 条，
    结果就是「搜什么都是同一批岗位」。这个坑踩过，所以在这里写清楚。

    另外**完整的方向名往往太长搜不到**（``Java后端开发`` 只有 1 条、``Java`` 有 225 条），
    所以要逐级放宽检索词并合并结果，见 ``_search_terms``。
    """

    kind = "public_api"
    SEARCH_API = "https://www.nowcoder.com/np-api/u/job/square-search"
    PAGE_SIZE = 50
    JD_LIMIT = 6000
    #: 每个检索词最多翻几页（词会换，所以不需要翻很多页）
    PAGES_PER_TERM = 1

    def _search_terms(self) -> list[str]:
        terms = [str(t).strip() for t in (self.config.get("_search_terms") or []) if str(t).strip()]
        if terms:
            return terms
        single = str(self.config.get("_search_keyword") or "").strip()
        return [single] if single else [""]

    def _build_payload(self, term: str, page: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "recruitType": int(self.config.get("recruit_type") or 1),
            "page": page,
            "pageSize": self.PAGE_SIZE,
        }
        if term:
            payload["query"] = term
        city = str(self.config.get("_city") or "").strip()
        if city:
            payload["city"] = city
        return payload

    @staticmethod
    def _company_of(item: dict[str, Any]) -> str:
        recommend = item.get("recommendInternCompany") or {}
        if isinstance(recommend, dict) and recommend.get("companyName"):
            return _clean(recommend.get("companyName"))
        identity = item.get("user") or {}
        if isinstance(identity, dict):
            for entry in identity.get("identity") or []:
                if isinstance(entry, dict) and entry.get("companyName"):
                    return _clean(entry.get("companyName"))
        return ""

    @staticmethod
    def _jd_of(item: dict[str, Any]) -> str:
        raw = item.get("ext")
        if not raw:
            return ""
        try:
            ext = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return ""
        parts = []
        if ext.get("infos"):
            parts.append("【岗位职责】" + _clean(ext["infos"]))
        if ext.get("requirements"):
            parts.append("【任职要求】" + _clean(ext["requirements"]))
        return "\n".join(parts)

    def _to_job(self, item: dict[str, Any]) -> JobItem | None:
        title = _clean(item.get("jobName"))
        job_id = item.get("id")
        if not title or job_id is None:
            return None

        salary_min = item.get("salaryMin")
        salary_max = item.get("salaryMax")
        salary_month = int(item.get("salaryMonth") or 12)
        salary_text = ""
        if salary_min and salary_max:
            suffix = f"·{salary_month}薪" if salary_month != 12 else ""
            salary_text = f"{salary_min:g}-{salary_max:g}K{suffix}"

        cities = item.get("jobCityList") or []
        city = " / ".join(str(c) for c in cities[:3]) if cities else _clean(item.get("jobCity"))

        tags: list[str] = []
        if item.get("jobKeys"):
            tags.extend(t.strip() for t in str(item["jobKeys"]).split(",") if t.strip())
        if item.get("graduationYear"):
            tags.append(_clean(item["graduationYear"]))
        recruit_type = item.get("recruitType")
        job_type = "校招" if recruit_type == 1 else "社招"
        tags.append(job_type)

        return self.make_job(
            title=title,
            city=city,
            company=self._company_of(item),
            job_type=job_type,
            url=f"https://www.nowcoder.com/jobs/detail/{job_id}",
            jd_text=self._jd_of(item)[: self.JD_LIMIT],
            salary_text=salary_text,
            published_at=_ms_to_day(item.get("refreshTime") or item.get("createTime")),
            tags=tags,
        )

    def fetch(self, budget: CrawlBudget) -> list[JobItem]:
        jobs: list[JobItem] = []
        seen: set[str] = set()
        max_jobs = int(self.config.get("max_jobs") or 400)
        terms = self._search_terms()

        for term in terms:
            for page in range(1, self.PAGES_PER_TERM + 1):
                if budget.remaining() <= 0 or len(jobs) >= max_jobs:
                    break
                try:
                    data = self.http_post_json(
                        self.SEARCH_API,
                        self._build_payload(term, page),
                        headers={"Referer": self.careers_url or "https://www.nowcoder.com/"},
                        timeout=25,
                    )
                except Exception as exc:  # noqa: BLE001 - 单个检索词失败不影响整体
                    logger.warning("[%s] 检索词 %r 第 %s 页失败：%s", self.label, term, page, exc)
                    continue

                body = (data or {}).get("data") or {}
                items = body.get("datas") or []
                if not items:
                    break

                for wrapper in items:
                    item = wrapper.get("data") if isinstance(wrapper, dict) else None
                    if not isinstance(item, dict):
                        continue
                    job_id = str(item.get("id"))
                    if job_id in seen:
                        continue
                    job = self._to_job(item)
                    if job is None:
                        continue
                    seen.add(job_id)
                    jobs.append(job)

                if len(items) < self.PAGE_SIZE:
                    break

        with_salary = sum(1 for job in jobs if job.salary_text)
        logger.info(
            "[%s] 用 %d 个检索词抓到 %d 个岗位（含薪资 %d 条）",
            self.label, len(terms), len(jobs), with_salary,
        )
        return jobs


class JsonApiCrawler(BaseCrawler):
    """**可配置**的 JSON 接口爬虫——公司官网招聘的主力。

    公司数量是无穷的，为每家写一个类不现实。绝大多数公司招聘站的岗位列表
    都是前端自己调的公开 JSON 接口，形态高度相似，所以这里做成「配置驱动」：
    换一家公司只改 ``sources.json``，不用改代码。

    已用此适配器验证通过的公司：

    * **腾讯**  ``careers.tencent.com``（公开 GET 接口，返回 ``Data.Posts``）
    * **网易**  ``hr.163.com``（POST，返回 ``data.list``）

    配置示例：:

        {
          "key": "tencent",
          "kind": "public_api",
          "careers_url": "https://careers.tencent.com/",
          "config": {
            "adapter": "json_api",
            "method": "GET",
            "endpoint": "https://careers.tencent.com/tencentcareer/api/post/Query",
            "params": {
              "keyword": "{keyword}", "pageIndex": "{page}", "pageSize": "50",
              "language": "zh-cn", "area": "cn", "timestamp": "{timestamp}"
            },
            "items_path": "Data.Posts",
            "field_map": {
              "title": "RecruitPostName", "city": "LocationName",
              "company": "ComName", "jd_text": "Responsibility",
              "url": "PostURL", "published_at": "LastUpdateTime",
              "experience": "RequireWorkYearsName"
            }
          }
        }

    占位符：``{keyword}`` ``{page}`` ``{city}`` ``{city_code}`` ``{timestamp}``。
    值为空的占位符会**连键一起删掉**，所以「城市参数没配置」不会发出
    ``cityId=`` 这种空参数，而是退化成全国检索，再由上层按城市过滤。

    ⚠️ 大多数公司官网**不公开薪资**（薪资是聚合站和社招平台才标的）。
    所以这类来源能提供权威的一手 JD 和岗位，但通常不贡献薪资样本，
    薪资仍主要靠牛客 / 51job 这类带薪资的来源。
    """

    kind = "public_api"

    @property
    def method(self) -> str:
        return str(self.config.get("method") or "GET").upper()

    @property
    def endpoint(self) -> str:
        return str(self.config.get("endpoint") or "")

    @property
    def field_map(self) -> dict[str, Any]:
        return dict(self.config.get("field_map") or {})

    @property
    def pages_per_term(self) -> int:
        return max(1, int(self.config.get("pages_per_term") or self.config.get("max_pages") or 1))

    @property
    def page_start(self) -> int:
        """首页页码。多数站点从 1 开始，少数（如京东）从 0 开始。"""
        return int(self.config.get("page_start") or 1)

    def _search_terms(self) -> list[str]:
        terms = [
            str(t).strip()
            for t in (self.config.get("_search_terms") or [])
            if str(t).strip()
        ]
        if terms:
            return terms[: int(self.config.get("max_terms") or 3)]
        single = str(self.config.get("_search_keyword") or "").strip()
        return [single] if single else [""]

    def _context(self, keyword: str, page: int) -> dict[str, Any]:
        return {
            "keyword": keyword,
            "page": page,
            "city": str(self.config.get("_city") or ""),
            "city_code": str(self.config.get("_city_code") or ""),
            "timestamp": int(time.time() * 1000),
        }

    @classmethod
    def _render(cls, node: Any, context: dict[str, Any]) -> Any:
        """填占位符，并保留原始类型。

        ``"{page}"`` 这种「整个值就是一个占位符」的情况会替换成**原始类型**
        （int 还是 int）——不少接口对 ``pageNo`` 要求是数字，传字符串会报错。
        渲染后为空字符串的键会被删掉。
        """
        if isinstance(node, str):
            for name, value in context.items():
                if node == "{" + name + "}":
                    return value
            rendered = node
            for name, value in context.items():
                rendered = rendered.replace("{" + name + "}", str(value))
            return rendered
        if isinstance(node, dict):
            result: dict[str, Any] = {}
            for key, value in node.items():
                filled = cls._render(value, context)
                if filled == "" or filled is None:
                    continue
                result[key] = filled
            return result
        if isinstance(node, list):
            return [cls._render(item, context) for item in node]
        return node

    def _request(self, keyword: str, page: int) -> Any:
        context = self._context(keyword, page)
        headers = {"Referer": self.careers_url or self.endpoint}
        if self.method == "GET":
            params = self._render(dict(self.config.get("params") or {}), context)
            response = self.http_get(
                self.endpoint, params=params, headers=headers, timeout=25
            )
            return _decode_response(response)
        body = self._render(dict(self.config.get("body") or {}), context)
        return self.http_post_json(self.endpoint, body, headers=headers, timeout=25)

    def fetch(self, budget: CrawlBudget) -> list[JobItem]:
        if not self.endpoint:
            raise ValueError("未配置 endpoint（岗位接口地址）")
        items_path = str(self.config.get("items_path") or "")
        if not items_path:
            raise ValueError("未配置 items_path（岗位数组的 JSON 路径）")

        jobs: list[JobItem] = []
        seen: set[str] = set()
        max_jobs = int(self.config.get("max_jobs") or 300)
        url_strip_query = bool(self.config.get("url_strip_query"))
        terms = self._search_terms()

        for term in terms:
            for offset in range(self.pages_per_term):
                page = self.page_start + offset
                if budget.remaining() <= 0 or len(jobs) >= max_jobs:
                    break
                try:
                    data = self._request(term, page)
                except Exception as exc:  # noqa: BLE001 - 单页失败不放弃整体
                    logger.warning("[%s] 词 %r 第 %s 页失败：%s", self.label, term, page, exc)
                    break

                items = json_path_get(data, items_path)
                if not isinstance(items, list) or not items:
                    break

                for raw in items:
                    if not isinstance(raw, dict):
                        continue
                    job = build_job_from_mapping(
                        self.make_job, raw, self.field_map, url_strip_query=url_strip_query
                    )
                    if job is None or job.job_key in seen:
                        continue
                    seen.add(job.job_key)
                    jobs.append(job)

                page_size = int(self.config.get("page_size") or len(items))
                if len(items) < page_size:
                    break

        with_salary = sum(1 for job in jobs if job.salary_text)
        logger.info(
            "[%s] 用 %d 个检索词抓到 %d 个岗位（含薪资 %d 条）",
            self.label, len(terms), len(jobs), with_salary,
        )
        return jobs


ADAPTERS: dict[str, type[BaseCrawler]] = {
    "jd": JDCampusCrawler,
    "jd_social": JDSocialCrawler,
    "nowcoder": NowcoderCrawler,
    "json_api": JsonApiCrawler,
}


def build(source: dict[str, Any]) -> BaseCrawler | None:
    adapter = str((source.get("config") or {}).get("adapter") or "")
    # 没写 adapter 时按 key 猜一把，减少配置负担
    if not adapter:
        adapter = str(source.get("key") or "").split("_")[0]
    cls = ADAPTERS.get(adapter)
    if cls is None:
        return None
    return cls(source)


__all__ = ["ADAPTERS", "JDCampusCrawler", "JDSocialCrawler", "JsonApiCrawler", "build"]
