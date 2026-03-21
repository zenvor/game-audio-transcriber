#!/usr/bin/env python3
"""根据 rename_plan.json 导出音频索引。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "category",
    "status",
    "original_name",
    "text",
    "sanitized_text",
    "requested_source_path",
    "source_path",
    "target_dir",
    "target_path",
    "source_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出音频索引清单")
    parser.add_argument(
        "--plan",
        default="output/rename_plan.json",
        help="重命名计划文件路径",
    )
    parser.add_argument(
        "--csv-out",
        default="output/audio_index.csv",
        help="导出的 CSV 索引路径",
    )
    parser.add_argument(
        "--json-out",
        default="output/audio_index.json",
        help="导出的 JSON 索引路径",
    )
    parser.add_argument(
        "--status",
        default="planned",
        help="只导出指定状态；传 all 表示导出全部",
    )
    return parser.parse_args()


def load_plan(plan_path: Path) -> list[dict]:
    with plan_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def filter_rows(rows: list[dict], status: str) -> list[dict]:
    if status == "all":
        return rows
    return [row for row in rows if row.get("status") == status]


def serialize_rows(rows: list[dict]) -> list[dict]:
    serialized: list[dict] = []
    for row in rows:
        serialized.append({field: row.get(field, "") for field in FIELDS})
    return serialized


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    plan_path = (project_root / args.plan).resolve()
    csv_path = (project_root / args.csv_out).resolve()
    json_path = (project_root / args.json_out).resolve()

    rows = load_plan(plan_path)
    rows = filter_rows(rows, args.status)
    rows = serialize_rows(rows)

    write_csv(rows, csv_path)
    write_json(rows, json_path)

    print(f"已导出 {len(rows)} 条索引")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
