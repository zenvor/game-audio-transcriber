"""
pipeline.py — 完整处理流水线
扫描 → 全量 Whisper 转写 → no_speech_prob 分流 → CLAP 音效分类
"""

from __future__ import annotations

import os
import json
import time
import gc
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


def release_cuda_memory(tag: str):
    """主动释放 PyTorch CUDA cache，降低 Whisper->CLAP 串行阶段的显存压力。"""
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"[显存回收] {tag}")
    except Exception:
        pass


# ── 幻觉词表加载 ──────────────────────────────────────────
import re

_FALLBACK_HALLUCINATION_TEXTS = {
    "thanks for watching", "thank you for watching",
    "transcription by eso", "translation by",
    "like and subscribe", "please subscribe",
    "subtitles by the amara org community", "satsang with mooji",
}

_HALLUCINATION_LEXICON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "whisper_hallucinations_en_zh.txt",
)


def _normalize_for_lexicon(text: str) -> str:
    """归一化文本：小写、去标点、压缩空白。加载词表和匹配时共用。"""
    normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _load_hallucination_lexicon() -> set[str]:
    try:
        with open(_HALLUCINATION_LEXICON_PATH, "r", encoding="utf-8") as f:
            lines = {_normalize_for_lexicon(line) for line in f if line.strip()}
        lines.discard("")
        if lines:
            print(f"[幻觉词表] 加载 {len(lines)} 条（{_HALLUCINATION_LEXICON_PATH}）")
            return lines
    except Exception as exc:
        print(f"[幻觉词表] 加载失败（{exc}），降级到内置词表")
    return set(_FALLBACK_HALLUCINATION_TEXTS)


_HALLUCINATION_TEXTS = _load_hallucination_lexicon()


def _is_hallucination(text: str) -> bool:
    return _normalize_for_lexicon(text) in _HALLUCINATION_TEXTS


def _is_extreme_repetition(text: str) -> bool:
    """
    检测极端重复文本（Whisper 在非语音输入上的典型幻觉模式）。
    """
    normalized = _normalize_for_lexicon(text)
    if not normalized or len(normalized) < 4:
        return False

    # 1. 单字符连续重复 >= 10（如 "rrrrrrrrrrr"）
    if re.search(r"(.)\1{9,}", normalized):
        return True

    # 2. 字符模式循环重复 >= 5 次（如 "huhuhuhuhuh"、"nanananana"）
    if re.search(r"(.{2,3})\1{4,}", normalized):
        return True

    words = normalized.split()

    # 3. 单词级连续重复 >= 4 次（如 "10 10 10 10"）
    if len(words) >= 4:
        streak = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                streak += 1
                if streak >= 4:
                    return True
            else:
                streak = 1

    # 4. n-gram 短语重复：任何 2~4 词组合出现 >= 2 次（如 "thank you thank you"）
    if len(words) >= 4:
        for n in range(2, min(5, len(words))):
            seen = set()
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                if gram in seen:
                    return True
                seen.add(gram)

    return False


# ── 人声/音效判定（多信号联合） ────────────────────────────────

def has_speech_result(result: dict) -> tuple[bool, str]:
    """
    多层判定，返回 (is_speech, reason_code)。
    R0a: 极端 compression_ratio (>= 5.0) → 音效
    R0b: 极端重复文本 + 统计信号异常 → 音效
    R1: nsp < 0.6 → 人声
    R2: 文本为空 → 音效
    R3: 文本命中幻觉词表 → 音效
    R4: 多信号补充拦截 → 音效
    R5: nsp < 0.85 且有文本 → 人声
    R6: 其余 → 音效
    """
    nsp = float(result["no_speech_prob"])
    text = str(result.get("text", "")).strip()
    cr = result.get("compression_ratio")
    alp = result.get("avg_logprob")

    # R0a: 极端 compression_ratio 独立拦截（如 cr=13.28，不论 nsp）
    if cr is not None and cr >= config.HALLUCINATION_EXTREME_COMPRESSION_RATIO:
        return False, "R0_EXTREME_CR"

    # R0b: 极端重复文本 + 至少一个统计信号异常 → 音效（覆盖 nsp）
    if text and _is_extreme_repetition(text):
        has_anomaly = (
            (cr is not None and cr >= config.HALLUCINATION_COMPRESSION_RATIO)
            or (alp is not None and alp <= config.HALLUCINATION_AVG_LOGPROB)
        )
        if has_anomaly:
            return False, "R0_EXTREME_REPETITION"

    # R1: 低 nsp 直接判人声
    if nsp < config.NO_SPEECH_THRESHOLD:
        return True, "R1_LOW_NSP"

    # R2: 无文本 → 音效
    if not text:
        return False, "R2_EMPTY_TEXT"

    # R3: 幻觉词表命中 → 音效
    if _is_hallucination(text):
        return False, "R3_HALLUCINATION_LEXICON"

    # R4: 多信号补充拦截（未收录的幻觉）
    if (cr is not None and cr >= config.HALLUCINATION_COMPRESSION_RATIO
            and alp is not None and alp <= config.HALLUCINATION_AVG_LOGPROB):
        return False, "R4_MULTI_SIGNAL"

    # R5: 有文本且 nsp 在灰区 → 人声
    if nsp < config.NO_SPEECH_THRESHOLD_WITH_TEXT:
        return True, "R5_TEXT_IN_GRAY_ZONE"

    # R6: nsp 很高 → 音效
    return False, "R6_HIGH_NSP"


def build_speech_result(path: str, result: dict) -> dict:
    return {
        "text": result["text"],
        "lang": result["lang"],
        "duration": result["duration"],
        "no_speech_prob": result["no_speech_prob"],
        "avg_logprob": result.get("avg_logprob"),
        "compression_ratio": result.get("compression_ratio"),
        "path": path,
    }


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


def recheck_sfx_results(output_dir: str, device: str | None = None):
    """重新检查已有 sfx_results.json，把误分流的人声迁回 results.json。"""
    start_time = time.time()
    output_path = os.path.join(output_dir, config.OUTPUT_FILE)
    sfx_out_path = os.path.join(output_dir, "sfx_results.json")

    existing_sfx = load_existing(sfx_out_path)
    if not existing_sfx:
        print(f"未找到已有音效结果: {sfx_out_path}")
        return

    speech_results = load_existing(output_path)
    updated_sfx = dict(existing_sfx)
    transcriber = Transcriber(device=device)

    migrated = 0
    kept = 0
    skipped = 0
    failed = 0
    total = len(existing_sfx)

    print(f"从 sfx_results.json 读取到 {total} 个音效文件，开始重检是否误分流...")

    for index, (filename, meta) in enumerate(existing_sfx.items(), start=1):
        path = str(meta.get("path", "")).strip()
        if not path:
            skipped += 1
            print(f"[{index}/{total}] {filename} | 跳过 | 缺少 path 字段")
            continue
        if not os.path.exists(path):
            skipped += 1
            print(f"[{index}/{total}] {filename} | 跳过 | 文件不存在: {path}")
            continue

        try:
            result = transcriber.transcribe(path)
        except Exception as exc:
            failed += 1
            print(f"[{index}/{total}] {filename} | 错误 | {exc}")
            continue

        no_speech_prob = result["no_speech_prob"]
        text = result["text"]
        is_speech, reason = has_speech_result(result)
        if is_speech:
            speech_results[filename] = build_speech_result(path, result)
            updated_sfx.pop(filename, None)
            migrated += 1
            tag = f"迁回语音({reason})"
        else:
            kept += 1
            tag = f"保留音效({reason})"

        print(
            f"[{index}/{total}] {filename[:30]:<30} "
            f"| {tag} | nsp={no_speech_prob:.2f} | {text[:30]}"
        )

    if migrated > 0:
        save_results(speech_results, output_path)
        save_results(updated_sfx, sfx_out_path)
    else:
        print("\n无需迁移，跳过写入。")

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"重检完成！迁回语音: {migrated} 个，保留音效: {kept} 个")
    print(f"跳过: {skipped} 个，失败: {failed} 个")
    print(f"results: {output_path}")
    print(f"sfx_results: {sfx_out_path}")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*50}")


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

    # 3. Silero VAD 预分流（可通过 config.USE_VAD_PREFILTER 关闭）
    sfx_candidates: list[str] = []
    todo_whisper = todo

    if config.USE_VAD_PREFILTER:
        try:
            from src.vad import SileroVAD
            vad = SileroVAD()
            todo_whisper = []
            print(f"Silero VAD 预扫描 {len(todo)} 个文件...")
            for j, path in enumerate(todo):
                if vad.has_speech(path):
                    todo_whisper.append(path)
                else:
                    sfx_candidates.append(path)
                if (j + 1) % 200 == 0:
                    print(f"  VAD 进度: {j+1}/{len(todo)}")
            print(
                f"VAD 完成：{len(todo_whisper)} 个待 Whisper，"
                f"{len(sfx_candidates)} 个直接归音效\n"
            )
        except Exception as exc:
            print(f"[VAD 初始化失败] {exc}，已回退：全量走 Whisper\n")
            todo_whisper = todo
            sfx_candidates = []

    # 4. Whisper 转写（仅处理 VAD 认定含人声的文件）
    transcriber = Transcriber(device=device)
    speech_results = dict(existing_speech)
    failed = []

    for i, path in enumerate(todo_whisper):
        filename = Path(path).name
        try:
            result = transcriber.transcribe(path)
            no_speech_prob = result["no_speech_prob"]
            text = result["text"]
            is_speech, reason = has_speech_result(result)

            if is_speech:
                # 有人声，保留转写结果
                speech_results[filename] = build_speech_result(path, result)
            else:
                # 纯音效，收集待 CLAP 分类
                sfx_candidates.append(path)

            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (len(todo_whisper) - i - 1) / speed if speed > 0 else 0
            tag = f"语音({reason})" if is_speech else f"音效({reason})"
            print(
                f"[{i+1}/{len(todo_whisper)}] {filename[:30]:<30} "
                f"| {tag} | nsp={no_speech_prob:.2f} | {result['text'][:30]} "
                f"| {speed:.1f}/s | 剩余 {eta/60:.1f}min"
            )

            # 每 500 个存一次，防止中途丢失
            if (i + 1) % 500 == 0:
                save_results(speech_results, output_path)

        except Exception as e:
            print(f"  [错误] {filename}: {e}", flush=True)
            failed.append({"path": path, "error": str(e)})

    # 5. 保存人声转写结果
    save_results(speech_results, output_path)

    # 进入 CLAP 前主动释放 Whisper 占用资源，避免同进程串行阶段显存挤占。
    del transcriber
    release_cuda_memory("Whisper 阶段结束，准备 CLAP")

    # 6. 对纯音效文件进行 CLAP 分类
    if sfx_candidates:
        print(f"\n音效文件 {len(sfx_candidates)} 个，开始 CLAP 分类...")
        from src.classifier import batch_classify
        sfx_new = batch_classify(
            sfx_candidates,
            checkpoint_path=sfx_out_path,
            existing_results=existing_sfx,
        )
        sfx_results = dict(existing_sfx)
        sfx_results.update(sfx_new)
        save_results(sfx_results, sfx_out_path)
        print(f"音效分类结果已保存 → {sfx_out_path}")

    # 7. 保存失败记录
    if failed:
        save_results(failed, failed_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"完成！人声转写 {len(speech_results)} 个，音效 {len(sfx_candidates)} 个")
    print(f"失败: {len(failed)} 个")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"结果: {output_path}")
    print(f"{'='*50}")
