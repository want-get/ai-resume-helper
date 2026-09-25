"""pytest 公共配置。

* 把项目根加入 sys.path
* **把数据目录指向测试专用目录**：否则自检会往用户的真实数据里写档案、建知识库
* 强制离线 Mock 模式（不调用真实大模型）

> 说明：系统临时目录在受限环境里可能不可写/不可删；本套测试刻意不依赖它。
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须在导入 paths / config **之前**设置：数据目录是在模块导入时解析的。
# 这样测试用的数据库、向量库、配置都在 .pytest_data/ 下，与用户数据完全隔离。
TEST_DATA_HOME = ROOT / ".pytest_data"
os.environ["AI_HELPER_DATA_DIR"] = str(TEST_DATA_HOME)

# ⚠️ 关键：.env 里的 DATABASE_URL 是相对路径（./data/app.db），
# 会覆盖上面的数据目录、把测试写进用户的真实数据库。
# 环境变量的优先级高于 .env，所以这里显式指到测试目录。
os.environ["DATABASE_URL"] = (
    f"sqlite+aiosqlite:///{(TEST_DATA_HOME / 'app.db').as_posix()}"
)
os.environ["CHROMA_DIR"] = str(TEST_DATA_HOME / "chroma")

# 单元测试不调用真实大模型，避免消耗额度、也保证结果稳定
os.environ.setdefault("LLM_MOCK_MODE", "true")
os.environ.setdefault("AUTO_SEED_DB", "true")
# 测试目录每次都重新建库，保证与真实数据无关
os.environ.setdefault("AUTO_BUILD_KB", "true")


@pytest.fixture()
def state_dir():
    """项目内的临时目录，替代 tmp_path。"""
    path = ROOT / ".pytest_data" / "artifacts" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
