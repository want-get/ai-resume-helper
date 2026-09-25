"""爬虫配置层测试：字段映射、URL 模板、来源清单合并、爬虫分派。

这些是「配置驱动」的部分——没有测试的话，一个占位符或路径写错，
只会表现成「某个来源莫名其妙抓不到东西」，很难定位。
"""

from __future__ import annotations

import json

import pytest

from backend.crawlers import build_crawler, is_crawlable
from backend.crawlers import base as base_mod
from backend.crawlers.base import BaseCrawler, CrawlBudget, DEFAULT_SOURCES
from backend.crawlers.fieldmap import (
    build_job_from_mapping,
    clean_url,
    json_path_get,
    ms_to_day,
    pick_field,
)
from backend.crawlers.public_api import JsonApiCrawler
from backend.crawlers.render_api import PlaywrightApiCrawler


# ---------------------------------------------------------------------- #
# json_path_get
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("data", "path", "expected"),
    [
        ({"a": {"b": {"c": 1}}}, "a.b.c", 1),
        ({"a": [{"name": "x"}, {"name": "y"}]}, "a.1.name", "y"),
        ({"a": [1, 2, 3]}, "a.-1", 3),
        ({"a": 1}, "a.b", None),
        ({"a": 1}, "nope", None),
        ({"a": 1}, "", {"a": 1}),
    ],
)
def test_json_path_get(data, path, expected):
    assert json_path_get(data, path) == expected


# ---------------------------------------------------------------------- #
# pick_field：四种取值写法
# ---------------------------------------------------------------------- #
def test_pick_field_plain_path():
    item = {"jobName": "会计"}
    assert pick_field(item, {"title": "jobName"}, "title") == "会计"


def test_pick_field_list_paths_joined():
    """网易的 JD 分成 requirement / description 两段，要拼起来。"""
    item = {"requirement": "岗位要求", "description": "岗位职责"}
    field = {"jd_text": ["requirement", "description"]}
    assert pick_field(item, field, "jd_text", join="\n") == "岗位要求\n岗位职责"


def test_pick_field_template_injects_json_path():
    """网易列表接口不给详情链接，用模板拼出来。"""
    item = {"id": 58384}
    field = {"url": "https://hr.163.com/job-detail.html?id={id}"}
    assert pick_field(item, field, "url") == "https://hr.163.com/job-detail.html?id=58384"


def test_pick_field_json_array_is_joined():
    """网易的 workPlaceNameList 是数组，应拼成字符串而不是列表字面量。"""
    item = {"workPlaceNameList": ["杭州市", "上海市"]}
    assert pick_field(item, {"city": "workPlaceNameList"}, "city") == "杭州市 / 上海市"


def test_pick_field_missing_returns_empty():
    assert pick_field({}, {"title": "jobName"}, "title") == ""
    assert pick_field({"a": 1}, {}, "title") == ""


# ---------------------------------------------------------------------- #
# build_job_from_mapping
# ---------------------------------------------------------------------- #
def _crawler(config: dict | None = None, label: str = "测试来源") -> BaseCrawler:
    return BaseCrawler({"key": "t", "label": label, "config": config or {}})


def test_mapping_prefers_numeric_salary_over_text():
    """数字薪资（元/月）比「6-8.5千」这种文案可靠，应优先。"""
    crawler = _crawler()
    job = build_job_from_mapping(
        crawler.make_job,
        {"jobName": "会计", "provideSalaryString": "6-8.5千",
         "jobSalaryMin": "6000", "jobSalaryMax": "8500"},
        {"title": "jobName", "salary_text": "provideSalaryString",
         "salary_min_yuan": "jobSalaryMin", "salary_max_yuan": "jobSalaryMax"},
    )
    assert job is not None
    assert job.salary_text == "6000-8500元/月"


def test_mapping_falls_back_to_salary_text():
    crawler = _crawler()
    job = build_job_from_mapping(
        crawler.make_job,
        {"jobName": "会计", "provideSalaryString": "6-8.5千"},
        {"title": "jobName", "salary_text": "provideSalaryString",
         "salary_min_yuan": "jobSalaryMin"},
    )
    assert job is not None and job.salary_text == "6-8.5千"


def test_mapping_requires_title():
    crawler = _crawler()
    assert build_job_from_mapping(crawler.make_job, {"x": 1}, {"title": "jobName"}) is None


def test_mapping_keeps_config_default_company_when_unmapped():
    """字段里没有 company 时，不能传空串——否则会覆盖配置里的默认公司名。"""
    crawler = _crawler({"company": "网易"})
    job = build_job_from_mapping(crawler.make_job, {"name": "后端"}, {"title": "name"})
    assert job is not None and job.company == "网易"


def test_mapping_uses_mapped_company_when_present():
    crawler = _crawler({"company": "网易"})
    job = build_job_from_mapping(
        crawler.make_job, {"name": "后端", "ComName": "腾讯云智"},
        {"title": "name", "company": "ComName"},
    )
    assert job is not None and job.company == "腾讯云智"


def test_mapping_ms_timestamp_and_extra_tags():
    crawler = _crawler()
    job = build_job_from_mapping(
        crawler.make_job,
        {"name": "后端", "updateTime": 1789480785000,
         "reqEducationName": "本科", "reqWorkYearsName": "不限",
         "jobTags": ["五险", "双休"]},
        {"title": "name", "published_at_ms": "updateTime", "tags": "jobTags",
         "degree": "reqEducationName", "experience": "reqWorkYearsName"},
    )
    assert job is not None
    assert job.published_at == ms_to_day(1789480785000)
    # 标签要合并「列表里的」和「附加字段」，且去重
    assert "五险" in job.tags and "本科" in job.tags and "不限" in job.tags


def test_ms_to_day_bad_input_is_empty():
    assert ms_to_day(None) == "" and ms_to_day("abc") == ""


# ---------------------------------------------------------------------- #
# clean_url
# ---------------------------------------------------------------------- #
def test_clean_url_strips_query_when_asked():
    """51job 的 req= 每次检索都变，不剥掉会产生重复岗位。"""
    url = "https://jobs.51job.com/a/1.html?s=x&req=abc"
    assert clean_url(url, strip_query=True) == "https://jobs.51job.com/a/1.html"


def test_clean_url_keeps_query_for_id_based_sites():
    """网易靠 ?id= 定位详情页，削掉就变成空页面了。"""
    url = "https://hr.163.com/job-detail.html?id=58384"
    assert clean_url(url, strip_query=False) == url


# ---------------------------------------------------------------------- #
# 来源清单合并
# ---------------------------------------------------------------------- #
def test_load_sources_adds_missing_builtins(tmp_path, monkeypatch):
    """老用户升级后，文件里没有 51job，应被补进来。"""
    path = tmp_path / "sources.json"
    monkeypatch.setattr(base_mod, "sources_file_path", lambda: path)
    path.write_text(
        json.dumps({"version": 1, "sources": [
            {"key": "nowcoder_campus", "label": "改过的名字", "enabled": False}
        ]}, ensure_ascii=False),
        encoding="utf-8",
    )

    sources = base_mod.load_sources()
    keys = [s["key"] for s in sources]

    assert "51job" in keys, "新增的内置来源应被补入"
    assert "tencent" in keys and "netease" in keys
    # 用户已有的条目必须原样保留（包括被禁用和改过的名字）
    existing = next(s for s in sources if s["key"] == "nowcoder_campus")
    assert existing["label"] == "改过的名字" and existing["enabled"] is False
    # 补入的结果应被写回文件，避免每次启动都重复合并
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "51job" in [s["key"] for s in saved["sources"]]


def test_load_sources_does_not_duplicate_existing(tmp_path, monkeypatch):
    path = tmp_path / "sources.json"
    monkeypatch.setattr(base_mod, "sources_file_path", lambda: path)
    path.write_text(
        json.dumps({"sources": [dict(s) for s in DEFAULT_SOURCES]}, ensure_ascii=False),
        encoding="utf-8",
    )
    sources = base_mod.load_sources()
    keys = [s["key"] for s in sources]
    assert len(keys) == len(set(keys)), "不应产生重复来源"


# ---------------------------------------------------------------------- #
# 可抓取判断 / 分派
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ({"kind": "public_api", "enabled": True}, True),
        ({"kind": "render_api", "enabled": True}, True),
        # 通用渲染爬虫必须填了地址才算可用
        ({"kind": "generic_render", "enabled": True, "config": {}}, False),
        ({"kind": "generic_render", "enabled": True, "config": {"list_url": "http://x"}}, True),
        ({"kind": "generic_render", "enabled": False, "config": {"list_url": "http://x"}}, False),
        ({"kind": "public_api", "enabled": False}, False),
    ],
)
def test_is_crawlable(source, expected):
    assert is_crawlable(source) is expected


@pytest.mark.parametrize(
    ("kind", "cls"),
    [("public_api", JsonApiCrawler), ("render_api", PlaywrightApiCrawler)],
)
def test_build_crawler_dispatch(kind, cls):
    source = {"key": "x", "kind": kind, "config": {"adapter": "json_api"}}
    assert isinstance(build_crawler(source), cls)


def test_build_crawler_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_crawler({"key": "x", "kind": "nope"})


# ---------------------------------------------------------------------- #
# render_api 的 URL 构造
# ---------------------------------------------------------------------- #
def _render_crawler(city: str = "北京", codes: dict | None = None) -> PlaywrightApiCrawler:
    crawler = PlaywrightApiCrawler({
        "key": "51job",
        "label": "51job",
        "kind": "render_api",
        "config": {
            "page_url": "https://we.51job.com/pc/search?keyword={keyword}&jobArea={city_code}",
            "intercept": ["/api/job/search-pc"],
            "items_path": "resultbody.job.items",
            "city_codes": codes if codes is not None else {"北京": "010000"},
        },
    })
    crawler.config["_city"] = city
    return crawler


def test_build_url_uses_verified_city_code():
    crawler = _render_crawler()
    assert crawler._city_code() == "010000"
    url = crawler._build_url("会计", 1)
    assert "jobArea=010000" in url and "keyword=%E4%BC%9A%E8%AE%A1" in url


def test_build_url_drops_param_when_city_unknown():
    """城市没配置代码时必须**整段删掉参数**，不能留下 jobArea= 空值。

    空 jobArea 会让站点返回不受控的结果；删掉它则退化为全国检索，
    再由上层按城市过滤，结果依然正确。
    """
    crawler = _render_crawler(city="未知小城", codes={"北京": "010000"})
    assert crawler._city_code() == ""
    url = crawler._build_url("会计", 1)
    assert "jobArea" not in url, url
    assert "keyword=" in url, "别的参数不能被一起删掉"


def test_build_url_drops_leading_param_and_restores_question_mark():
    crawler = PlaywrightApiCrawler({
        "key": "x", "label": "x", "kind": "render_api",
        "config": {
            "page_url": "https://e.com/search?jobArea={city_code}&keyword={keyword}",
            "intercept": ["/a"], "items_path": "a.b", "city_codes": {},
        },
    })
    crawler.config["_city"] = "未知"
    url = crawler._build_url("会计", 1)
    assert url.startswith("https://e.com/search?keyword="), url


def test_build_url_matches_city_with_district_suffix():
    """「北京·朝阳区」也要能匹配到北京的代码。"""
    crawler = _render_crawler(city="北京·朝阳区", codes={"北京": "010000"})
    assert crawler._city_code() == "010000"


def test_51job_default_source_has_no_page_num_in_url():
    """踩过的坑：pageNum 拼进搜索页 URL 不会真的翻页，必须靠点按钮。"""
    source = next(s for s in DEFAULT_SOURCES if s["key"] == "51job")
    assert "pageNum" not in source["config"]["page_url"]
    assert source["config"]["next_selector"], "必须配置翻页按钮选择器"


def test_default_sources_all_have_unique_keys_and_valid_kind():
    keys = [s["key"] for s in DEFAULT_SOURCES]
    assert len(keys) == len(set(keys))
    for source in DEFAULT_SOURCES:
        assert source["kind"] in {"public_api", "render_api", "generic_render"}
