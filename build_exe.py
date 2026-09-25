"""把项目打包成单文件 exe（PyInstaller）。

    python build_exe.py              # 打 onefile（双击即用，单个文件）
    python build_exe.py --onedir     # 打 onedir（启动更快，多个文件）
    python build_exe.py --clean      # 先清掉旧的 build/dist

要点说明
--------
* **不打包 Chromium**：爬虫用 ``channel="msedge"`` 驱动系统自带的 Edge/Chrome，
  exe 里不含浏览器，体积能小 150MB+。
* **不打包 Streamlit**：前端是 ``web/`` 下的静态单页应用，由 FastAPI 直接托管。
* **向量化是离线的**：用内置的字符 n-gram 哈希 TF-IDF，不需要下载模型，
  打包后也不会因为联网下载失败而跑不起来。
* 只把 ``web/`` 与 ``data/seed/`` 作为**数据文件**打进去；``rag/`` ``backend/``
  ``prompts/`` 是 Python 包，作为代码一起编译。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "AI求职助手"
ENTRY = ROOT / "main.py"

# PyInstaller 有时看不到这些「动态导入」的模块，必须显式声明
HIDDEN_IMPORTS = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "aiosqlite",
    "sqlalchemy.dialects.sqlite",
    "sqlalchemy.dialects.sqlite.aiosqlite",
    "sqlalchemy.dialects.mysql",
    "sqlalchemy.dialects.mysql.aiomysql",
    "aiomysql",
    "backend.app_v3",
    "backend.crawlers.public_api",
    "backend.crawlers.generic_render",
    "playwright",
    "playwright.sync_api",
    "bs4",
    "lxml",
    "lxml.etree",
    "multipart",
    "multipart.multipart",
]

# 明显用不到、体积又大的东西，排除掉能显著减小体积
EXCLUDES = [
    "streamlit",          # v3 前端已改为静态单页
    "matplotlib", "pandas", "scipy", "sklearn",
    "tkinter", "PyQt5", "PySide2", "PySide6",
    "IPython", "jupyter", "notebook",
    "pytest", "setuptools", "pip",
]

# 需要连数据一起收集的包（内含非代码资源）
COLLECT_ALL = ["chromadb"]

DATAS = [
    (ROOT / "web", "web"),
    (ROOT / "data" / "seed", "data/seed"),
    (ROOT / "prompts", "prompts"),
]


def build(mode: str, clean: bool, console: bool) -> int:
    if clean:
        for name in ("build", "dist"):
            target = ROOT / name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                print(f"已清理 {target}")

    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", APP_NAME,
        "--onefile" if mode == "onefile" else "--onedir",
        "--console" if console else "--windowed",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
    ]
    for source, dest in DATAS:
        if not source.exists():
            print(f"[警告] 数据目录不存在，跳过：{source}")
            continue
        args += ["--add-data", f"{source}{__import__('os').pathsep}{dest}"]
    for name in HIDDEN_IMPORTS:
        args += ["--hidden-import", name]
    for name in COLLECT_ALL:
        args += ["--collect-all", name]
    for name in EXCLUDES:
        args += ["--exclude-module", name]

    args.append(str(ENTRY))

    print("=" * 74)
    print(f"  开始打包（{mode}）")
    print("=" * 74)
    print("  " + " ".join(args[:8]) + " ...")
    print()

    started = time.time()
    result = subprocess.run(args, cwd=str(ROOT))
    elapsed = time.time() - started

    if result.returncode != 0:
        print(f"\n[失败] PyInstaller 返回 {result.returncode}")
        return result.returncode

    target = ROOT / "dist" / (f"{APP_NAME}.exe" if mode == "onefile" else APP_NAME)
    print("\n" + "=" * 74)
    print(f"  打包完成，用时 {elapsed / 60:.1f} 分钟")
    print("=" * 74)
    if target.exists():
        if target.is_file():
            size = target.stat().st_size / 1024 / 1024
            print(f"  产物：{target}")
            print(f"  大小：{size:.1f} MB")
        else:
            total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            print(f"  产物目录：{target}")
            print(f"  总大小　：{total / 1024 / 1024:.1f} MB")
            print(f"  启动文件：{target / (APP_NAME + '.exe')}")
    print()
    print("  使用：双击该 exe，浏览器会自动打开界面。")
    print("  数据会写在 exe 同级的 data/ 目录下（若该目录不可写则用 %LOCALAPPDATA%）。")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="打包成 exe")
    parser.add_argument("--onedir", action="store_true", help="打成目录（启动更快）")
    parser.add_argument("--clean", action="store_true", help="先清理 build/dist")
    parser.add_argument("--windowed", action="store_true",
                        help="不显示控制台窗口（出问题时不易排查，不建议）")
    options = parser.parse_args()
    sys.exit(build("onedir" if options.onedir else "onefile",
                   options.clean, console=not options.windowed))
