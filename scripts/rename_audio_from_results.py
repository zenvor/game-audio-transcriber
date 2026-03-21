#!/usr/bin/env python3
"""根据 results.json / sfx_results.json 批量分类并重命名音频文件。"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path


INVALID_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WHITESPACE_PATTERN = re.compile(r"\s+")
SEPARATOR_PATTERN = re.compile(r"[_-]{2,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 output 结果文件生成或执行音频分类重命名"
    )
    parser.add_argument(
        "--results",
        default="output/results.json",
        help="语音结果文件路径",
    )
    parser.add_argument(
        "--sfx-results",
        default="output/sfx_results.json",
        help="音效结果文件路径",
    )
    parser.add_argument(
        "--plan-out",
        default="output/rename_plan.json",
        help="导出的重命名计划文件",
    )
    parser.add_argument(
        "--target-root",
        default="renamed_audio",
        help="分类重命名后的输出根目录",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行复制重命名；默认只预览和导出计划",
    )
    return parser.parse_args()


def normalize_source_path(raw_path: str, project_root: Path) -> Path:
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return (project_root / normalized).resolve()


def resolve_existing_source(source_path: Path) -> Path | None:
    if source_path.exists():
        return source_path

    pattern = f"*__{source_path.stem}{source_path.suffix.lower()}"
    matches = sorted(source_path.parent.glob(pattern))
    if matches:
        return matches[0]

    return None


def sanitize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = INVALID_CHARS_PATTERN.sub("_", text)
    text = WHITESPACE_PATTERN.sub("_", text)
    text = text.replace("、", "_")
    text = text.replace("，", "_")
    text = text.replace("。", "")
    text = text.replace("！", "")
    text = text.replace("？", "")
    text = text.replace("：", "_")
    text = text.replace("；", "_")
    text = text.replace("（", "_")
    text = text.replace("）", "_")
    text = text.replace("(", "_")
    text = text.replace(")", "_")
    text = text.replace("[", "_")
    text = text.replace("]", "_")
    text = text.replace("{", "_")
    text = text.replace("}", "_")
    text = text.replace("+", "_")
    text = text.replace("&", "_")
    text = text.replace(",", "_")
    text = text.replace(".", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace("__", "_")
    text = SEPARATOR_PATTERN.sub("_", text)
    return text.strip("._-")


def iter_entries(json_path: Path) -> list[tuple[str, dict]]:
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data.items())


def category_name(json_path: Path) -> str:
    return "speech" if json_path.name == "results.json" else "sfx"


def relative_source_dir(source_path: Path, input_root: Path) -> Path:
    try:
        return source_path.parent.relative_to(input_root)
    except ValueError:
        return Path()


def build_plan(project_root: Path, json_paths: list[Path], target_root: Path) -> list[dict]:
    plan: list[dict] = []
    reserved_targets: dict[Path, Path] = {}
    input_root = (project_root / "input").resolve()

    for json_path in json_paths:
        category = category_name(json_path)
        for original_name, meta in iter_entries(json_path):
            raw_text = (meta.get("text") or "").strip()
            raw_path = meta.get("path") or ""

            if not raw_path:
                plan.append(
                    {
                        "source_json": str(json_path),
                        "original_name": original_name,
                        "status": "skipped_missing_path",
                    }
                )
                continue

            requested_source_path = normalize_source_path(raw_path, project_root)
            source_path = resolve_existing_source(requested_source_path)
            if source_path is None:
                plan.append(
                    {
                        "source_json": str(json_path),
                        "original_name": original_name,
                        "source_path": str(requested_source_path),
                        "status": "skipped_source_not_found",
                    }
                )
                continue

            sanitized = sanitize_text(raw_text)
            if not sanitized:
                plan.append(
                    {
                        "source_json": str(json_path),
                        "original_name": original_name,
                        "source_path": str(source_path),
                        "status": "skipped_empty_text",
                    }
                )
                continue

            original_stem = Path(original_name).stem
            target_name = f"{sanitized}__{original_stem}{source_path.suffix.lower()}"
            relative_dir = relative_source_dir(source_path, input_root)
            target_dir = target_root / category / relative_dir
            target_path = target_dir / target_name

            item = {
                "category": category,
                "source_json": str(json_path),
                "original_name": original_name,
                "requested_source_path": str(requested_source_path),
                "source_path": str(source_path),
                "target_path": str(target_path),
                "target_dir": str(target_dir),
                "text": raw_text,
                "sanitized_text": sanitized,
            }

            existing_owner = reserved_targets.get(target_path)
            if existing_owner and existing_owner != source_path:
                item["status"] = "conflict_duplicate_target"
                plan.append(item)
                continue

            if target_path.exists() and target_path != source_path:
                item["status"] = "conflict_target_exists"
                plan.append(item)
                continue

            reserved_targets[target_path] = source_path
            item["status"] = "planned"
            plan.append(item)

    return plan


def save_plan(plan: list[dict], plan_path: Path) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)


def print_summary(plan: list[dict]) -> None:
    summary: dict[str, int] = {}
    category_summary: dict[str, int] = {}
    for item in plan:
        status = item["status"]
        summary[status] = summary.get(status, 0) + 1
        category = item.get("category")
        if category and status == "planned":
            category_summary[category] = category_summary.get(category, 0) + 1

    print("分类重命名计划统计:")
    for status in sorted(summary):
        print(f"  {status}: {summary[status]}")
    if category_summary:
        print("分类数量:")
        for category in sorted(category_summary):
            print(f"  {category}: {category_summary[category]}")

    preview = [item for item in plan if item["status"] == "planned"][:15]
    if preview:
        print("\n预览前 15 条 planned 项：")
        for item in preview:
            print(f"  {item['source_path']} -> {item['target_path']}")


def apply_plan(plan: list[dict]) -> int:
    copied = 0
    for item in plan:
        if item["status"] != "planned":
            continue
        source = Path(item["source_path"])
        target = Path(item["target_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    json_paths = [
        (project_root / args.results).resolve(),
        (project_root / args.sfx_results).resolve(),
    ]
    plan_path = (project_root / args.plan_out).resolve()
    target_root = (project_root / args.target_root).resolve()

    plan = build_plan(project_root, json_paths, target_root)
    save_plan(plan, plan_path)
    print_summary(plan)
    print(f"\n重命名计划已写入: {plan_path}")
    print(f"目标根目录: {target_root}")

    if not args.apply:
        print("\n当前为 dry-run；确认无误后加 --apply 再执行实际分类复制。")
        return 0

    conflicts = [
        item for item in plan
        if item["status"].startswith("conflict_")
    ]
    if conflicts:
        print("\n存在冲突项，已停止执行。请先检查 rename_plan.json。")
        return 1

    copied = apply_plan(plan)
    print(f"\n已完成分类复制: {copied} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
