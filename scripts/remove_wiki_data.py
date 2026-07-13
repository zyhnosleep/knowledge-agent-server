"""Remove legacy Wiki data after an explicit operator confirmation.

This script is intentionally separate from application startup. It creates a
SQLite backup, drops the legacy wiki_pages table, and removes the legacy data/wiki
directory only when --confirm is supplied.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/app.db"))
    parser.add_argument("--wiki-dir", type=Path, default=Path("data/wiki"))
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("refusing to delete Wiki data without --confirm")
    if not args.database.exists():
        raise SystemExit(f"database not found: {args.database}")

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup = args.database.with_name(f"{args.database.stem}.pre-wiki-removal-{stamp}.db")
    shutil.copy2(args.database, backup)
    with sqlite3.connect(args.database) as connection:
        connection.execute("DROP TABLE IF EXISTS wiki_pages")
        connection.commit()
    if args.wiki_dir.exists():
        shutil.rmtree(args.wiki_dir)
    print(f"backup: {backup}")
    print("removed: wiki_pages and legacy Wiki directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
