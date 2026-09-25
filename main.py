"""AI 求职助手 —— 程序入口（单机版）。

双击 exe 或运行 ``python main.py`` 会：

    1. 在本地找一个空闲端口启动后端（默认 127.0.0.1:8000）
    2. 自动打开浏览器进入界面
    3. 保持运行，直到你按 Ctrl+C 或关闭窗口

**不需要安装 MySQL、不需要改配置文件、不需要另开前端进程。**
数据（数据库、向量库、日志、配置）都写在程序同级目录的 ``data/`` 下。
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# 打包成 exe 后工作目录可能是任意位置，先把可执行文件所在目录塞进 sys.path
if getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(sys.executable).resolve().parent))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import describe as describe_paths  # noqa: E402

BANNER = r"""
    _    ___    ___  _  _ ___ ___    _   _ ___ _    ___  ___
   /_\  |_ _|  | _ \| || | __| _ \  | | | | __| |  | _ \| _ \
  / _ \  | |   |   /| __ | _||   /  | |_| | _|| |__|  _/|   /
 /_/ \_\|___|  |_|_\|_||_|___|_|_\   \___/|___|____|_|  |_|_\
                                        AI 求职助手 · 单机版
"""


def find_free_port(preferred: int, limit: int = 20) -> int:
    """从 preferred 开始向后找一个能绑定的端口。"""
    for offset in range(max(1, limit)):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"从 {preferred} 起连续 {limit} 个端口都被占用，请先关掉占用的程序")


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.6)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_until_ready(port: int, timeout: float = 90.0) -> bool:
    """等后端真正可用（首次运行要建库，可能要十几秒）。"""
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if response.status_code == 200:
                return True
        except Exception:  # noqa: BLE001 - 还没起来，继续等
            time.sleep(0.4)
    return False


def run_server(host: str, port: int, log_level: str) -> None:
    import uvicorn

    uvicorn.run(
        "backend.app_v3:app",
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AI 求职助手（单机版）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=None, help="监听端口")
    parser.add_argument("--no-browser", action="store_true", help="不要自动打开浏览器")
    parser.add_argument("--reload", action="store_true", help="开发模式：改代码自动重启")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    args = parser.parse_args()

    from config import get_settings

    settings = get_settings()
    preferred = args.port or settings.port

    print(BANNER)
    print("=" * 66)
    print(f"  版本　　：{settings.app_version}")
    print(f"  数据目录：{describe_paths()['data_home']}")
    print("  数据库　：SQLite（本地文件，无需安装任何数据库）")
    print("=" * 66)

    # 已经在跑就直接开浏览器，避免重复启动
    if port_in_use(preferred) and wait_until_ready(preferred, timeout=3):
        url = f"http://127.0.0.1:{preferred}"
        print(f"\n  检测到程序已经在运行：{url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    port = find_free_port(preferred, limit=settings.port_scan_limit)
    url = f"http://127.0.0.1:{port}"
    if port != preferred:
        print(f"\n  端口 {preferred} 被占用，改用 {port}")

    if args.reload:
        print(f"\n  开发模式已启动：{url}\n")
        if not args.no_browser:
            threading.Timer(2.0, lambda: webbrowser.open(url)).start()
        run_server(args.host, port, args.log_level)
        return 0

    # 后台线程起服务，主线程等就绪并打开浏览器
    thread = threading.Thread(
        target=run_server, args=(args.host, port, args.log_level), daemon=True
    )
    thread.start()

    print("\n  正在启动服务（首次运行要建库，可能需要十几秒）...")
    if not wait_until_ready(port):
        print("\n  [错误] 服务启动超时。日志见数据目录下的 logs/app.log")
        return 1

    print(f"\n  [OK] 已就绪，界面地址：{url}")
    print("  " + "-" * 62)
    print("  使用步骤：① 填求职档案并上传简历  ->  ② 抓取岗位与薪资")
    print("            ③ 生成专属知识库        ->  ④ 开始模拟面试")
    print("  " + "-" * 62)
    print("  按 Ctrl+C 退出（或直接关闭本窗口）\n")

    if settings.open_browser and not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001 - 打不开浏览器不影响使用
            print(f"  （自动打开浏览器失败：{exc}，请手动访问上面的地址）")

    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  正在退出...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
