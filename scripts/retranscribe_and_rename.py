#!/usr/bin/env python3
"""对已重命名的音频文件重新跑 Whisper 转录，并根据新结果原地重命名。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 让 import config / src.* 正常工作
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.transcriber import Transcriber
from scripts.rename_audio_from_results import sanitize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对指定音频文件重新 Whisper 转录并原地重命名"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="音频文件或目录（目录会递归扫描 .wav）",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--model", default=None, help="覆盖 Whisper 模型")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览，不实际重命名",
    )
    return parser.parse_args()


def collect_files(paths: list[str]) -> list[Path]:
    """从参数列表收集所有音频文件。"""
    result: list[Path] = []
    for p in paths:
        path = Path(p).resolve()
        if path.is_file():
            if path.suffix.lower() in config.SUPPORTED_FORMATS:
                result.append(path)
            else:
                print(f"跳过不支持的格式: {path}")
        elif path.is_dir():
            for root, _, names in os.walk(path):
                for name in names:
                    if name.lower().endswith(config.SUPPORTED_FORMATS):
                        result.append(Path(root, name).resolve())
        else:
            print(f"路径不存在: {path}")
    return sorted(set(result))


def extract_original_stem(filename: str) -> str:
    """从 '<text>__<original_stem>.ext' 中提取 original_stem。"""
    stem = Path(filename).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def main() -> int:
    args = parse_args()

    if args.model:
        config.WHISPER_MODEL = args.model

    files = collect_files(args.paths)
    if not files:
        print("未找到任何音频文件")
        return 1

    print(f"共 {len(files)} 个音频文件待处理")
    if args.dry_run:
        print("** dry-run 模式，不会实际重命名 **\n")

    transcriber = Transcriber(device=args.device)

    renamed = 0
    skipped = 0
    failed = 0
    unchanged = 0
    reserved_targets: set[Path] = set()
    start = time.time()

    for i, filepath in enumerate(files, 1):
        old_name = filepath.name
        try:
            result = transcriber.transcribe(str(filepath))
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(files)}] {old_name} | 转录失败 | {exc}")
            continue

        new_text = result["text"]
        sanitized = sanitize_text(new_text)
        if not sanitized:
            skipped += 1
            print(
                f"[{i}/{len(files)}] {old_name} | 跳过 | "
                f"转录文本为空 (nsp={result['no_speech_prob']:.2f})"
            )
            continue

        original_stem = extract_original_stem(old_name)
        ext = filepath.suffix
        new_name = f"{sanitized}__{original_stem}{ext}"
        new_path = filepath.parent / new_name

        if new_path == filepath:
            unchanged += 1
            print(f"[{i}/{len(files)}] {old_name} | 无变化")
            continue

        if new_path in reserved_targets or (new_path.exists() and new_path != filepath):
            skipped += 1
            print(
                f"[{i}/{len(files)}] {old_name} | 跳过 | "
                f"目标已存在: {new_name}"
            )
            continue

        reserved_targets.add(new_path)

        if not args.dry_run:
            try:
                os.rename(filepath, new_path)
            except OSError as exc:
                failed += 1
                print(f"[{i}/{len(files)}] {old_name} | 重命名失败 | {exc}")
                continue

        renamed += 1
        tag = "将重命名" if args.dry_run else "已重命名"
        print(f"[{i}/{len(files)}] {tag}: {old_name} -> {new_name}")

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"完成！重命名: {renamed}, 无变化: {unchanged}, 跳过: {skipped}, 失败: {failed}")
    print(f"耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*50}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
