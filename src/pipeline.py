"""
pipeline.py — 完整处理流水线
扫描 → 全量 Whisper 转写 → no_speech_prob 分流 → CLAP 音效分类
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path

import config
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


def run_sfx_only(output_dir: str):
    """仅对已有 sfx_results.json 中的音效文件重新跑 CLAP 分类"""
    start_time = time.time()
    sfx_out_path = os.path.join(output_dir, "sfx_results.json")

    existing_sfx = load_existing(sfx_out_path)
    if not existing_sfx:
        print(f"未找到已有音效结果: {sfx_out_path}")
        return

    # 从已有结果中提取文件路径
    sfx_files = [entry["path"] for entry in existing_sfx.values() if "path" in entry]
    print(f"从 sfx_results.json 读取到 {len(sfx_files)} 个音效文件，开始 CLAP 重新分类...")

    from src.classifier import batch_classify
    sfx_results = batch_classify(sfx_files)
    save_results(sfx_results, sfx_out_path)

    elapsed = time.time() - start_time
    print(f"\n完成！重新分类 {len(sfx_results)} 个音效文件")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"结果: {sfx_out_path}")


def run(input_dir: str, output_dir: str, device: str | None = None):
    start_time = time.time()
    output_path = os.path.join(output_dir, config.OUTPUT_FILE)
    sfx_out_path = os.path.join(output_dir, "sfx_results.json")
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

    # 2. 断点续跑：加载已有结果
    existing_speech = load_existing(output_path)
    existing_sfx = load_existing(sfx_out_path)
    already_done = set(existing_speech.keys()) | set(existing_sfx.keys())
    todo = [f for f in all_files if Path(f).name not in already_done]
    print(f"待处理: {len(todo)} 个（已完成: {len(already_done)} 个）\n")

    if not todo:
        print("所有文件已处理完毕！")
        return

    # 3. 全量 Whisper 转写
    transcriber = Transcriber(device=device)
    speech_results = dict(existing_speech)
    sfx_candidates = []
    failed = []

    for i, path in enumerate(todo):
        filename = Path(path).name
        try:
            result = transcriber.transcribe(path)
            no_speech_prob = result["no_speech_prob"]

            if no_speech_prob < config.NO_SPEECH_THRESHOLD:
                # 有人声，保留转写结果
                speech_results[filename] = {
                    "text": result["text"],
                    "lang": result["lang"],
                    "duration": result["duration"],
                    "no_speech_prob": no_speech_prob,
                    "path": path,
                }
            else:
                # 纯音效，收集待 CLAP 分类
                sfx_candidates.append(path)

            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / speed if speed > 0 else 0
            tag = "语音" if no_speech_prob < config.NO_SPEECH_THRESHOLD else "音效"
            print(
                f"[{i+1}/{len(todo)}] {filename[:30]:<30} "
                f"| {tag} | nsp={no_speech_prob:.2f} | {result['text'][:30]} "
                f"| {speed:.1f}/s | 剩余 {eta/60:.1f}min"
            )

            # 每 500 个存一次，防止中途丢失
            if (i + 1) % 500 == 0:
                save_results(speech_results, output_path)

        except Exception as e:
            print(f"  [错误] {filename}: {e}", flush=True)
            failed.append({"path": path, "error": str(e)})

    # 4. 保存人声转写结果
    save_results(speech_results, output_path)

    # 5. 对纯音效文件进行 CLAP 分类
    if sfx_candidates:
        print(f"\n音效文件 {len(sfx_candidates)} 个，开始 CLAP 分类...")
        from src.classifier import batch_classify
        sfx_new = batch_classify(sfx_candidates)
        sfx_results = dict(existing_sfx)
        sfx_results.update(sfx_new)
        save_results(sfx_results, sfx_out_path)
        print(f"音效分类结果已保存 → {sfx_out_path}")

    # 6. 保存失败记录
    if failed:
        save_results(failed, failed_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"完成！人声转写 {len(speech_results)} 个，音效 {len(sfx_candidates)} 个")
    print(f"失败: {len(failed)} 个")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"结果: {output_path}")
    print(f"{'='*50}")
