"""初始化数据库：建库建表 + 导入种子数据。

    python scripts/init_db.py                 # 建表 + 导入岗位/薪资种子数据
    python scripts/init_db.py --reset         # 先删除已有数据再导入
    python scripts/init_db.py --ddl           # 额外打印建表 DDL（便于查看 MySQL 表结构）

默认读取 ``.env`` 中的 ``DATABASE_URL``（MySQL + aiomysql）；
MySQL 不可用时会自动回退到 SQLite，并在输出中明确提示。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SEED_DIR, get_settings  # noqa: E402
from backend.db import dispose_database, init_database, session_scope  # noqa: E402
from backend.db.models import Base, JobPosting, SalaryRecord  # noqa: E402
from backend.db import repository  # noqa: E402
from rag import load_jsonl  # noqa: E402


async def main(reset: bool, show_ddl: bool) -> None:
    settings = get_settings()
    info = await init_database()
    print("=" * 68)
    print(f"数据库后端：{info['backend']}")
    print(f"连接串　　：{info['url']}")
    if info.get("fallback"):
        print("⚠️  MySQL 不可用，已回退到 SQLite（数据仍可用，但不是生产形态）")
        for err in info.get("errors", []):
            print(f"   - {err}")
    print("=" * 68)

    if show_ddl:
        from sqlalchemy.schema import CreateTable
        from sqlalchemy.dialects import mysql

        print("\n--- MySQL 建表 DDL ---")
        for table in Base.metadata.sorted_tables:
            print(str(CreateTable(table).compile(dialect=mysql.dialect())).strip() + ";\n")

    jobs = load_jsonl(SEED_DIR / "job_postings.jsonl")
    salaries = load_jsonl(SEED_DIR / "salary_records.jsonl")

    async with session_scope() as db:
        if reset:
            from sqlalchemy import delete

            await db.execute(delete(JobPosting))
            await db.execute(delete(SalaryRecord))
            print("已清空 job_postings / salary_records")

        existing_jobs = await repository.count_job_postings(db)
        existing_salaries = await repository.count_salary_records(db)

        if existing_jobs and not reset:
            print(f"岗位表已有 {existing_jobs} 条数据，跳过导入（如需重导请加 --reset）")
        else:
            count = await repository.upsert_job_postings(db, jobs)
            print(f"✅ 导入岗位 {count} 条")

        if existing_salaries and not reset:
            print(f"薪资表已有 {existing_salaries} 条数据，跳过导入（如需重导请加 --reset）")
        else:
            count = await repository.upsert_salary_records(db, salaries)
            print(f"✅ 导入薪资样本 {count} 条")

    async with session_scope() as db:
        print(f"\n当前数据量：岗位 {await repository.count_job_postings(db)} 条，"
              f"薪资 {await repository.count_salary_records(db)} 条")

    await dispose_database()
    print("\n完成。启动后端：python server.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="初始化数据库并导入种子数据")
    parser.add_argument("--reset", action="store_true", help="先清空再导入")
    parser.add_argument("--ddl", action="store_true", help="打印建表 DDL")
    args = parser.parse_args()
    asyncio.run(main(args.reset, args.ddl))
