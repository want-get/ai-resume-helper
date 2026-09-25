"""开发用后端启动脚本（不带前端启动器逻辑）。

平时直接用 ``python main.py`` 即可 —— 它会启动服务并自动打开浏览器。
这个脚本留给「只想跑后端、不想自动开浏览器」的场景。

    python server.py                 # http://127.0.0.1:8000/docs
    python server.py --port 8900 --reload
"""

from __future__ import annotations

import argparse

from config import get_settings
from paths import describe as describe_paths


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="启动 AI 求职助手后端（FastAPI）")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="改代码自动重启（开发用）")
    parser.add_argument(
        "--log-level", default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    args = parser.parse_args()

    import uvicorn

    paths = describe_paths()
    print("=" * 68)
    print(f"  {settings.app_name} v{settings.app_version}")
    print(f"  界面　　：http://{args.host}:{args.port}")
    print(f"  接口文档：http://{args.host}:{args.port}/docs")
    print(f"  数据目录：{paths['data_home']}")
    print("=" * 68)

    uvicorn.run(
        "backend.app_v3:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
