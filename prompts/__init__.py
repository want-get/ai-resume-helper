# prompts 包：按岗位类型组织简历助手的 Prompt 模板
#
# 目录结构：
#   common.py    公用规则（面试规则、评分权重、简历优化通用要求）
#   tech.py      技术岗模板
#   non_tech.py  非技术岗模板
#
# 使用方式：
#   from prompts import get_prompts, TECH
#   prompts = get_prompts(TECH)
#   prompt = prompts.RESUME_OPTIMIZE_PROMPT.format(resume_text="...")

from . import common, tech, non_tech

# 岗位类型常量
TECH = "tech"
NON_TECH = "non_tech"

# 岗位类型 -> 中文名（供界面展示）
ROLE_LABELS = {
    TECH: "技术岗",
    NON_TECH: "非技术岗",
}

# 岗位类型 -> prompt 模板模块
_PROMPT_MODULES = {
    TECH: tech,
    NON_TECH: non_tech,
}


def get_prompts(role_type):
    """根据岗位类型返回对应的 prompt 模板模块

    参数：
        role_type: prompts.TECH 或 prompts.NON_TECH

    返回：
        对应岗位的 prompt 模块，含以下四个模板：
        - RESUME_OPTIMIZE_PROMPT       简历优化
        - INTERVIEW_QUESTIONS_PROMPT   生成面试题
        - MOCK_INTERVIEW_SYSTEM_PROMPT 模拟面试
        - RESUME_SCORE_PROMPT          简历评分
    """
    module = _PROMPT_MODULES.get(role_type)
    if module is None:
        raise ValueError(f"未知的岗位类型：{role_type}")
    return module