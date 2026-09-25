"""统一的路径解析：兼容「源码运行」与「PyInstaller 打包成 exe 运行」两种形态。

打包后的关键差异：

* ``sys.frozen`` 为 True；代码/静态资源被解压到 ``sys._MEIPASS``（临时目录，
  退出即删），**不能往那里写数据**。
* 用户数据（SQLite、Chroma、日志、配置）必须落在 **exe 所在目录** 下，
  否则用户重启程序就丢了。

因此约定：

    DATA_HOME = <可写数据根目录>
        ├── app.db            SQLite 数据库
        ├── chroma/           Chroma 向量库
        ├── uploads/          上传的简历
        ├── logs/             日志
        └── settings.json     用户在前端保存的大模型配置

* 源码运行：``<项目根>/data``
* exe 运行：优先 ``<exe 同级>/data``；若该目录不可写（例如装在 Program Files），
  退回到 ``%LOCALAPPDATA%/AIResumeHelper/data``。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包出的可执行文件里。"""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """只读资源根目录（打包后是 _MEIPASS，否则是项目根）。

    只用来定位随程序分发的只读资源：``web/`` 静态页面、``data/seed/`` 种子数据、
    ``prompts/`` 等。**不要往这里写文件。**
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def executable_dir() -> Path:
    """可执行文件所在目录（源码运行时就是项目根）。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _resolve_data_home() -> Path:
    """决定可写数据根目录。允许用环境变量覆盖（测试与多实例用）。"""
    override = os.environ.get("AI_HELPER_DATA_DIR", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    candidates = [executable_dir() / "data"]
    if is_frozen():
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if local:
            candidates.append(Path(local) / "AIResumeHelper" / "data")

    for candidate in candidates:
        if _is_writable(candidate):
            return candidate
    # 兜底：用户主目录
    fallback = Path.home() / ".ai_resume_helper" / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# 只读资源
BUNDLE_ROOT: Path = bundle_root()
PROJECT_ROOT: Path = BUNDLE_ROOT          # 兼容旧引用
SEED_DIR: Path = BUNDLE_ROOT / "data" / "seed"
WEB_DIR: Path = BUNDLE_ROOT / "web"

# 可写数据
DATA_HOME: Path = _resolve_data_home()
CHROMA_DIR: Path = DATA_HOME / "chroma"
UPLOAD_DIR: Path = DATA_HOME / "uploads"
LOG_DIR: Path = DATA_HOME / "logs"
SETTINGS_FILE: Path = DATA_HOME / "settings.json"
DB_FILE: Path = DATA_HOME / "app.db"


def ensure_dirs() -> None:
    for path in (DATA_HOME, CHROMA_DIR, UPLOAD_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def describe() -> dict[str, str]:
    """给 /health 用：让用户一眼看出程序把数据写到哪了。"""
    return {
        "mode": "frozen(exe)" if is_frozen() else "source",
        "bundle_root": str(BUNDLE_ROOT),
        "executable_dir": str(executable_dir()),
        "data_home": str(DATA_HOME),
        "database": str(DB_FILE),
        "chroma": str(CHROMA_DIR),
        "settings_file": str(SETTINGS_FILE),
    }


ensure_dirs()
