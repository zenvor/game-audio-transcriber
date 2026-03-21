"""
pipeline.py — 完整处理流水线
扫描 → VAD 过滤 → Whisper 转写 → 输出 JSON
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path

import config
from src.vad_filter import filter_files
from src.transcriber import Transcriber


def scan_files(input_dir: str) -> list[str]:
    """递归扫描所有支持的音频文件"""
    files = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            if name.lower().endswith(config.SUPPORTED_FORMATS):
                files.append(os.path.join(root, name))
    return sorted(files)


def load_existing(output_path: str) -> dict:
    """加载已有结果，支持断点续跑"""
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run(input_dir: str, output_dir: str, device: str | None = None, skip_vad: bool = False):
    start_time = time.time()
    output_path = os.path.join(output_dir, config.OUTPUT_FILE)
    failed_path = os.path.join(output_dir, config.FAILED_FILE)

    print(f"\n{'='*50}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*50}\n")

    # 1. 扫描文件
    all_files = scan_files(input_dir)
    print(f"扫描完成，共 {len(all_files)} 个音频文件")

    if not all_files:
        print("没有找到音频文件，请检查 input 目录")
        return

    # 2. 工作文件（WAV 无需格式转换）
    work_files = all_files

    # 3. VAD 过滤
    if skip_vad:
        speech_files = work_files
        print(f"跳过 VAD，直接处理 {len(speech_files)} 个文件")
    else:
        speech_files, sfx_files = filter_files(
            work_files,
            threshold=config.VAD_THRESHOLD,
            min_speech_sec=config.VAD_MIN_SPEECH
        )
    # 3.5 对纯音效文件进行 YAMNet 分类
    if not skip_vad and sfx_files:
        from src.classifier import batch_classify
        sfx_results = batch_classify(sfx_files)
        sfx_out = os.path.join(output_dir, "sfx_results.json")
        save_results(sfx_results, sfx_out)
        print(f"音效分类结果已保存 → {sfx_out}")

    # 4. 断点续跑：跳过已处理文件
    existing = load_existing(output_path)
    todo = [f for f in speech_files if Path(f).name not in existing]
    print(f"\n待转写: {len(todo)} 个（已完成: {len(existing)} 个）\n")

    if not todo:
        print("所有文件已处理完毕！")
        return

    # 5. 批量转写
    transcriber = Transcriber(device=device)
    results = dict(existing)
    failed = []

    for i, path in enumerate(todo):
        filename = Path(path).name
        try:
            result = transcriber.transcribe(path)
            results[filename] = {**result, "path": path}

            if (i + 1) % config.LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                eta = (len(todo) - i - 1) / speed
                print(
                    f"[{i+1}/{len(todo)}] {filename[:30]:<30} "
                    f"| {result['lang']} | {result['text'][:25]} "
                    f"| 速度 {speed:.1f}个/s | 预计剩余 {eta/60:.1f}min"
                )

            # 每 500 个存一次，防止中途丢失
            if (i + 1) % 500 == 0:
                save_results(results, output_path)

        except Exception as e:
            failed.append({"path": path, "error": str(e)})
            if len(failed) % 20 == 0:
                print(f"  [警告] 已失败 {len(failed)} 个")

    # 6. 保存最终结果
    save_results(results, output_path)
    if failed:
        save_results(failed, failed_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"完成！共转写 {len(results)} 个文件")
    print(f"失败: {len(failed)} 个 → {failed_path}")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"结果: {output_path}")
    print(f"{'='*50}")
