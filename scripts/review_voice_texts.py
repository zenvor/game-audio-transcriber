#!/usr/bin/env python3
"""使用兼容 OpenAI 或 Gemini 接口的模型对语音识别文本做上下文纠错。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib import error, request


SYSTEM_PROMPT = """你是游戏本地化与语音资产整理助手。

现在需要你校正《王者荣耀》语音备注文本。当前处理的语音包、音频文件、目录命名和播报内容都与《王者荣耀》相关。输入文本由 Whisper large-v3 转写得到。Whisper large-v3 的常见问题不是凭空编造整句，而是把发音相近的字词、近音词、同音词、英文单词或短语识别错，也可能出现轻微标点偏差。

规则：
1. 输出必须是 JSON 对象，字段固定为 corrected_text、changed、reason、kind、needs_manual_review。
2. corrected_text 必须是适合做语音资源备注的短文本，不要扩写剧情，不要添加解释。
3. 优先结合《王者荣耀》常见播报、战场信号、术语、文件上下文和相邻条目进行修正，候选短语只是辅助参考，不是必须命中才能修正。
4. 如果原文已经基本正确，保留原意并尽量少改。
5. 如果无法确定，不要强行脑补，尽量保守修正。
6. 英文播报保持常见游戏播报写法；中文播报保持自然、简洁、符合游戏语境。
7. 不要无故删减信息。像 "Blue Team Rampage" 这类包含阵营信息的完整播报，除非确定有错，否则不能简化成 "Rampage!"。
8. 默认假设 ASR 错误主要表现为错字、错词、近音词替换，而不是凭空多出或少掉整段内容。除非你有非常强的上下文证据，否则不要删除阵营前缀、核心播报片段或完整短语。
9. 像 "Blue Team, Killing Spree." 这类文本，不能只因为候选短语里有 "Killing Spree!" 就擅自删掉 "Blue Team"；遇到这种情况应保留原文或标记人工复核。
10. 如果 changed=false，则 corrected_text 应与原文保持一致或只做极轻微标点修复。
11. kind 只能取以下枚举之一：asr_error、punctuation、normalization、uncertain、no_change。
12. needs_manual_review 只在你无法高置信判断，或认为仍需人工过目时设为 true。
13. reason 必须使用简体中文，禁止输出英文解释。
14. 要判断当前文本读起来是否自然、是否符合《王者荣耀》语境。如果识别结果像“小心对方偷他”这样读起来不合理，应优先考虑同音字、近音词或固定播报误识别，并修正为更合理的表达，例如“小心对方偷塔”。
15. 如果提供了前文和后文，要优先利用这些相邻条目判断当前播报是否连贯、是否属于固定播报链。
16. 判断时要优先考虑这是对“同一句口令/播报”的纠错：结合上下文、单词拼写、固定术语和整句是否通顺，判断当前文本里是否存在错别字或错误单词，而不是改写成另一句意思更短的播报。
"""

PROVIDER_PRESETS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-3-flash-preview",
        "api_key_env": "GEMINI_API_KEY",
        "api_style": "gemini",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
        "api_style": "openai",
    },
}

NATURAL_CHUNK_PATTERN = re.compile(r"(\d+)")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+|[\u4e00-\u9fff]")


class RequestFailure(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 OpenAI 或 Gemini 纠正语音文本")
    parser.add_argument("--input", default="output/results.json", help="语音结果文件")
    parser.add_argument("--manual-phrases", default="data/honor_of_kings_phrases.txt", help="人工维护短语词表")
    parser.add_argument("--provider", choices=sorted(PROVIDER_PRESETS), default="gemini", help="接口提供商预设")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容接口地址，默认使用 provider 预设")
    parser.add_argument("--model", default=None, help="模型名称，默认使用 provider 预设")
    parser.add_argument("--api-key-env", default=None, help="API key 对应的环境变量名，默认使用 provider 预设")
    parser.add_argument("--force", action="store_true", help="即使已有 reviewed 结果也重新处理")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条，便于验证")
    parser.add_argument("--min-frequency", type=int, default=2, help="自动抽取候选短语的最小频次")
    parser.add_argument("--candidate-limit", type=int, default=20, help="每条记录附带的候选短语数量")
    parser.add_argument("--context-window", type=int, default=2, help="每侧附带的相邻上下文条数")
    parser.add_argument("--batch-size", type=int, default=5, help="每批请求处理的条目数")
    parser.add_argument("--save-every", type=int, default=10, help="每处理 N 条写回一次结果")
    parser.add_argument("--max-retries", type=int, default=3, help="单条请求最大重试次数")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="重试基础退避秒数")
    parser.add_argument("--dry-run", action="store_true", help="只预览将处理的记录和候选短语，不调用 API")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def resolve_provider_config(args: argparse.Namespace) -> tuple[str, str, str, str]:
    preset = PROVIDER_PRESETS[args.provider]
    base_url = args.base_url or preset["base_url"]
    model = args.model or preset["model"]
    api_key_env = args.api_key_env or preset["api_key_env"]
    api_style = preset["api_style"]
    return base_url, model, api_key_env, api_style


def load_manual_phrases(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def normalize_phrase(text: str) -> str:
    return " ".join(text.strip().split())


def auto_extract_phrases(results: dict, min_frequency: int) -> list[str]:
    counter: Counter[str] = Counter()
    for meta in results.values():
        text = normalize_phrase(str(meta.get("text", "")))
        if not text:
            continue
        counter[text] += 1
    phrases = [
        phrase for phrase, count in counter.most_common()
        if count >= min_frequency and len(phrase) <= 40
    ]
    return phrases


def merge_phrase_candidates(results: dict, manual_path: Path, min_frequency: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for phrase in load_manual_phrases(manual_path) + auto_extract_phrases(results, min_frequency):
        normalized = normalize_phrase(phrase)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def similarity_score(current: str, candidate: str) -> float:
    current_lower = current.lower()
    candidate_lower = candidate.lower()
    seq_ratio = SequenceMatcher(None, current_lower, candidate_lower).ratio()
    overlap = len(set(current_lower) & set(candidate_lower))
    return seq_ratio + overlap * 0.01


def select_candidate_phrases(text: str, phrases: list[str], limit: int) -> list[str]:
    if not phrases:
        return []
    scored = sorted(
        ((similarity_score(text, phrase), phrase) for phrase in phrases),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = [phrase for score, phrase in scored if score > 0.15][:limit]
    # 有意不兜底：无相关候选时返回空列表，配合 prompt 中"不要求必须命中候选短语"的指令，
    # 让模型独立判断而非被无关短语误导。
    return selected


def normalize_context_path(raw_path: str, fallback_name: str) -> PurePosixPath:
    normalized = raw_path.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        normalized = fallback_name
    return PurePosixPath(normalized)


def natural_sort_key(text: str) -> list[object]:
    parts = NATURAL_CHUNK_PATTERN.split(text.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def entry_sort_key(filename: str, meta: dict) -> tuple[object, ...]:
    relative_path = normalize_context_path(str(meta.get("path", "")), filename)
    return (
        tuple(natural_sort_key(part) for part in relative_path.parent.parts),
        natural_sort_key(relative_path.name),
    )


def format_context_entry(filename: str, meta: dict) -> str:
    text = normalize_phrase(str(meta.get("text", ""))) or "（空）"
    relative_path = normalize_context_path(str(meta.get("path", "")), filename)
    return f"{relative_path.as_posix()} | {text}"


def build_neighbor_context(
    results: dict,
    context_window: int,
) -> dict[str, dict[str, list[str]]]:
    entries = list(results.items())
    contexts: dict[str, dict[str, list[str]]] = {
        filename: {"previous": [], "next": []}
        for filename, _ in entries
    }
    if context_window <= 0:
        return contexts

    grouped: dict[str, list[tuple[str, dict]]] = {}
    for filename, meta in entries:
        relative_path = normalize_context_path(str(meta.get("path", "")), filename)
        grouped.setdefault(relative_path.parent.as_posix(), []).append((filename, meta))

    for group_entries in grouped.values():
        ordered_group = sorted(group_entries, key=lambda item: entry_sort_key(item[0], item[1]))
        for index, (filename, _) in enumerate(ordered_group):
            previous_slice = ordered_group[max(0, index - context_window):index]
            next_slice = ordered_group[index + 1:index + 1 + context_window]
            contexts[filename]["previous"] = [
                format_context_entry(prev_filename, prev_meta)
                for prev_filename, prev_meta in previous_slice
            ]
            contexts[filename]["next"] = [
                format_context_entry(next_filename, next_meta)
                for next_filename, next_meta in next_slice
            ]

    # 回退补充：同目录邻居不足时，按 results dict 的插入顺序取相邻条目。
    # 前提：results.json 通常按目录/文件名有序写入；若顺序混乱，回退上下文
    # 可能语义不连续，但不影响正确性，只是参考价值降低。
    for index, (filename, _) in enumerate(entries):
        previous_items = contexts[filename]["previous"]
        next_items = contexts[filename]["next"]
        if len(previous_items) < context_window:
            start = max(0, index - context_window)
            for prev_filename, prev_meta in entries[start:index]:
                candidate = format_context_entry(prev_filename, prev_meta)
                if candidate not in previous_items:
                    previous_items.append(candidate)
                if len(previous_items) >= context_window:
                    break
        if len(next_items) < context_window:
            for next_filename, next_meta in entries[index + 1:index + 1 + context_window]:
                candidate = format_context_entry(next_filename, next_meta)
                if candidate not in next_items:
                    next_items.append(candidate)
                if len(next_items) >= context_window:
                    break

    for filename in contexts:
        contexts[filename]["previous"] = contexts[filename]["previous"][-context_window:]
        contexts[filename]["next"] = contexts[filename]["next"][:context_window]

    return contexts


def should_skip(meta: dict, force: bool) -> bool:
    if force:
        return False
    return (
        meta.get("correction_status") == "reviewed"
        and "correction_changed" in meta
        and "correction_kind" in meta
        and "needs_manual_review" in meta
    )


def build_request_item(
    filename: str,
    meta: dict,
    phrases: list[str],
    contexts: dict[str, dict[str, list[str]]],
    candidate_limit: int,
) -> dict:
    text = normalize_phrase(str(meta.get("text", "")))
    lang = str(meta.get("lang", ""))
    path = str(meta.get("path", ""))
    candidates = select_candidate_phrases(text, phrases, candidate_limit)
    context = contexts.get(filename, {"previous": [], "next": []})
    return {
        "filename": filename,
        "meta": meta,
        "text": text,
        "lang": lang,
        "path": path,
        "candidates": candidates,
        "previous_context": context["previous"],
        "next_context": context["next"],
    }


def build_messages(batch_items: list[dict]) -> list[dict]:
    item_blocks: list[str] = []
    for index, item in enumerate(batch_items, start=1):
        candidate_block = "\n".join(f"- {candidate}" for candidate in item["candidates"]) if item["candidates"] else "- 无"
        previous_block = "\n".join(f"- {entry}" for entry in item["previous_context"]) if item["previous_context"] else "- 无"
        next_block = "\n".join(f"- {entry}" for entry in item["next_context"]) if item["next_context"] else "- 无"
        item_blocks.append(
            f"""### 条目 {index}
filename: {item["filename"]}
语言: {item["lang"] or 'unknown'}
文件路径: {item["path"]}
当前识别文本: {item["text"]}

前文:
{previous_block}

后文:
{next_block}

候选短语:
{candidate_block}"""
        )

    joined_items = "\n\n".join(item_blocks)
    user_prompt = f"""请批量校正下面这些《王者荣耀》语音备注文本。

{joined_items}

请优先根据《王者荣耀》语境、相邻条目之间的连续关系，以及每句话本身读起来是否自然来判断。即使没有命中候选短语，也要独立判断当前识别文本是否合理；如果像错别字、同音字、近音词或固定播报误识别，应主动修正。

请严格输出 JSON 对象，顶层结构固定为：
{{
  "items": [
    {{
      "filename": "与输入完全一致的文件名",
      "corrected_text": "校正后的文本",
      "changed": true,
      "reason": "使用简体中文简短说明为什么这么改",
      "kind": "asr_error",
      "needs_manual_review": false
    }}
  ]
}}

要求：
1. items 数量必须与输入条目数量完全一致。
2. 每个 filename 必须与输入完全一致，不得省略、重命名或合并条目。
3. 可以调整 items 的顺序，但 filename 必须能唯一对应回输入。
4. 不要输出额外解释，不要输出 Markdown，只输出 JSON。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_openai_compatible(base_url: str, api_key: str, model: str, messages: list[dict]) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        retryable = exc.code == 429 or exc.code >= 500
        raise RequestFailure(f"HTTP {exc.code}: {detail}", retryable=retryable) from exc
    except error.URLError as exc:
        raise RequestFailure(f"网络错误: {exc}", retryable=True) from exc

    content = payload["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RequestFailure(f"模型返回非 JSON 内容: {content}", retryable=False) from exc


def call_gemini(base_url: str, api_key: str, model: str, messages: list[dict]) -> dict:
    system_text = "\n".join(item["content"] for item in messages if item["role"] == "system")
    user_text = "\n\n".join(item["content"] for item in messages if item["role"] == "user")
    body = {
        "systemInstruction": {
            "parts": [{"text": system_text}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/models/{model}:generateContent",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        retryable = exc.code == 429 or exc.code >= 500
        raise RequestFailure(f"HTTP {exc.code}: {detail}", retryable=retryable) from exc
    except error.URLError as exc:
        raise RequestFailure(f"网络错误: {exc}", retryable=True) from exc

    try:
        content = payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RequestFailure(f"Gemini 返回结构异常: {payload}", retryable=False) from exc
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RequestFailure(f"模型返回非 JSON 内容: {content}", retryable=False) from exc


def call_provider_once(api_style: str, base_url: str, api_key: str, model: str, messages: list[dict]) -> dict:
    if api_style == "gemini":
        return call_gemini(base_url=base_url, api_key=api_key, model=model, messages=messages)
    return call_openai_compatible(base_url=base_url, api_key=api_key, model=model, messages=messages)


def call_provider_with_retries(
    api_style: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_retries: int,
    retry_backoff: float,
) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            return call_provider_once(
                api_style=api_style,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
            )
        except RequestFailure as exc:
            if not exc.retryable or attempt == max_retries:
                raise
            time.sleep(retry_backoff * (2 ** (attempt - 1)))
    raise RequestFailure("请求失败且未返回结果", retryable=False)


def parse_batch_reviews(payload: dict, expected_filenames: list[str]) -> dict[str, dict]:
    if not isinstance(payload, dict):
        raise RequestFailure(f"模型返回顶层不是对象: {payload!r}", retryable=False)

    items = payload.get("items")
    if not isinstance(items, list):
        raise RequestFailure(f"模型返回缺少 items 数组: {payload!r}", retryable=False)
    if len(items) != len(expected_filenames):
        raise RequestFailure(
            f"模型返回条目数不匹配: 期望 {len(expected_filenames)} 条，实际 {len(items)} 条",
            retryable=False,
        )

    expected_set = set(expected_filenames)
    seen: set[str] = set()
    reviews_by_filename: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise RequestFailure(f"items 内存在非对象条目: {item!r}", retryable=False)
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename:
            raise RequestFailure(f"items 内存在缺失 filename 的条目: {item!r}", retryable=False)
        if filename not in expected_set:
            raise RequestFailure(f"模型返回了未知 filename: {filename}", retryable=False)
        if filename in seen:
            raise RequestFailure(f"模型返回了重复 filename: {filename}", retryable=False)
        seen.add(filename)
        reviews_by_filename[filename] = item

    missing = [filename for filename in expected_filenames if filename not in seen]
    if missing:
        raise RequestFailure(f"模型缺少以下 filename: {', '.join(missing)}", retryable=False)
    return reviews_by_filename


def chunk_queue(queue: list[tuple[str, dict]], batch_size: int) -> list[list[tuple[str, dict]]]:
    safe_batch_size = max(1, batch_size)
    return [queue[index:index + safe_batch_size] for index in range(0, len(queue), safe_batch_size)]


VALID_CORRECTION_KINDS = {
    "asr_error",
    "punctuation",
    "normalization",
    "uncertain",
    "no_change",
}

KIND_ALIASES = {
    "asr": "asr_error",
    "asrerror": "asr_error",
    "transcription_error": "asr_error",
    "识别错误": "asr_error",
    "asr识别错误": "asr_error",
    "标点": "punctuation",
    "标点修正": "punctuation",
    "punctuation_fix": "punctuation",
    "格式规范化": "normalization",
    "规范化": "normalization",
    "normalisation": "normalization",
    "unclear": "uncertain",
    "不确定": "uncertain",
    "需人工复核": "uncertain",
    "nochange": "no_change",
    "不修改": "no_change",
    "保持原文": "no_change",
}

PUNCTUATION_CHARS = " \t\r\n,.;:!?，。！？：；、'\"()[]{}<>-_/\\"


def punctuation_signature(text: str) -> str:
    return "".join(ch for ch in text if ch not in PUNCTUATION_CHARS)


def normalization_signature(text: str) -> str:
    return normalize_phrase(text).lower()


def normalize_correction_kind(raw_kind: object) -> str | None:
    if not raw_kind:
        return None
    normalized = normalize_phrase(str(raw_kind)).lower().replace(" ", "_")
    normalized = KIND_ALIASES.get(normalized, normalized)
    if normalized in VALID_CORRECTION_KINDS:
        return normalized
    return None


def infer_correction_kind(original: str, corrected: str, changed: bool) -> str:
    if not changed or original == corrected:
        return "no_change"
    if punctuation_signature(original) == punctuation_signature(corrected):
        return "punctuation"
    if normalization_signature(original) == normalization_signature(corrected):
        return "normalization"
    return "asr_error"


def normalize_manual_review(raw_value: object, correction_kind: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return correction_kind == "uncertain"
    normalized = normalize_phrase(str(raw_value)).lower()
    return normalized in {"1", "true", "yes", "y", "是", "需要", "需人工复核"}


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def extract_core_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_phrase(text).lower())


def deleted_block_matches_neighbor(
    source_tokens: list[str],
    start: int,
    end: int,
) -> bool:
    deleted = source_tokens[start:end]
    if not deleted:
        return False
    block_size = len(deleted)
    previous_block = source_tokens[max(0, start - block_size):start]
    next_block = source_tokens[end:end + block_size]
    return deleted == previous_block or deleted == next_block


def is_benign_repeated_block_cleanup(original: str, corrected: str) -> bool:
    original_tokens = extract_core_tokens(original)
    corrected_tokens = extract_core_tokens(corrected)
    if not original_tokens or not corrected_tokens or len(corrected_tokens) >= len(original_tokens):
        return False

    matcher = SequenceMatcher(None, original_tokens, corrected_tokens)
    saw_delete = False
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag != "delete" or not deleted_block_matches_neighbor(original_tokens, i1, i2):
            return False
        saw_delete = True
    return saw_delete


def is_strict_subsequence(candidate: list[str], source: list[str]) -> bool:
    if not candidate or len(candidate) >= len(source):
        return False
    source_index = 0
    for token in candidate:
        while source_index < len(source) and source[source_index] != token:
            source_index += 1
        if source_index >= len(source):
            return False
        source_index += 1
    return True


def is_suspicious_shortening(original: str, corrected: str) -> bool:
    if not original or not corrected or original == corrected:
        return False
    if punctuation_signature(original) == punctuation_signature(corrected):
        return False
    if normalization_signature(original) == normalization_signature(corrected):
        return False
    if is_benign_repeated_block_cleanup(original, corrected):
        return False

    original_tokens = extract_core_tokens(original)
    corrected_tokens = extract_core_tokens(corrected)
    if corrected_tokens and is_strict_subsequence(corrected_tokens, original_tokens):
        return True

    original_core = punctuation_signature(original).replace(" ", "").lower()
    corrected_core = punctuation_signature(corrected).replace(" ", "").lower()
    if (
        corrected_core
        and original_core
        and corrected_core in original_core
        and len(corrected_core) < len(original_core)
    ):
        return True

    return False


def fallback_reason_in_chinese(
    original: str,
    corrected: str,
    changed: bool,
    correction_kind: str,
    needs_manual_review: bool,
) -> str:
    if needs_manual_review or correction_kind == "uncertain":
        return "模型无法高置信判断，建议人工复核。"
    if correction_kind == "asr_error":
        return f"ASR 识别有误，已将“{original}”修正为更符合游戏语境的“{corrected}”。"
    if correction_kind == "punctuation":
        return "原文内容基本正确，仅做了标点或轻微格式修正。"
    if correction_kind == "normalization":
        return "原文语义不变，仅做了写法规范化处理。"
    if changed:
        return "已按游戏语境对原文做保守修正。"
    return "原文识别结果符合游戏语境，无需修改。"


def apply_conservative_guardrails(
    original: str,
    corrected: str,
    needs_manual_review: bool,
    reason: str,
) -> tuple[str, bool, str]:
    if not is_suspicious_shortening(original, corrected):
        return corrected, needs_manual_review, reason

    guardrail_reason = (
        f"模型给出的修正“{corrected}”存在缩句式删减，可能误删原文有效信息；"
        "已保留原文并标记为人工复核。"
    )
    return original, True, guardrail_reason


def resolve_correction_changed(original: str, corrected: str) -> bool:
    return original != corrected


def backfill_structured_review_fields(meta: dict) -> bool:
    if meta.get("correction_status") != "reviewed":
        return False
    if (
        "correction_changed" in meta
        and "correction_kind" in meta
        and "needs_manual_review" in meta
    ):
        return False

    original = normalize_phrase(str(meta.get("text", "")))
    corrected = normalize_phrase(str(meta.get("corrected_text", ""))) or original
    changed = resolve_correction_changed(original, corrected)
    correction_kind = infer_correction_kind(
        original=original,
        corrected=corrected,
        changed=changed,
    )

    meta["correction_changed"] = changed
    meta["correction_kind"] = correction_kind
    meta["needs_manual_review"] = correction_kind == "uncertain"
    return True


def apply_review(meta: dict, review: dict, model: str) -> None:
    original = normalize_phrase(str(meta.get("text", "")))
    corrected = normalize_phrase(str(review.get("corrected_text", ""))) or original
    requested_kind = normalize_correction_kind(review.get("kind"))
    initial_changed = resolve_correction_changed(original, corrected)
    if not initial_changed:
        correction_kind = "no_change"
    elif requested_kind and requested_kind != "no_change":
        correction_kind = requested_kind
    else:
        correction_kind = infer_correction_kind(
            original=original,
            corrected=corrected,
            changed=initial_changed,
        )
    needs_manual_review = normalize_manual_review(
        review.get("needs_manual_review"),
        correction_kind,
    )
    raw_reason = normalize_phrase(str(review.get("reason", "")))
    reason = raw_reason if contains_cjk(raw_reason) else fallback_reason_in_chinese(
        original=original,
        corrected=corrected,
        changed=initial_changed,
        correction_kind=correction_kind,
        needs_manual_review=needs_manual_review,
    )
    corrected, needs_manual_review, reason = apply_conservative_guardrails(
        original=original,
        corrected=corrected,
        needs_manual_review=needs_manual_review,
        reason=reason,
    )
    changed = resolve_correction_changed(original, corrected)
    if not changed:
        correction_kind = "no_change"

    meta["corrected_text"] = corrected
    meta["correction_changed"] = changed
    meta["correction_kind"] = correction_kind
    meta["needs_manual_review"] = needs_manual_review
    meta["correction_status"] = "reviewed"
    meta["correction_reason"] = reason
    meta["correction_model"] = model


def mark_error(meta: dict, model: str, message: str) -> None:
    meta["correction_status"] = "error"
    meta["correction_reason"] = message
    meta["correction_model"] = model


def print_review_log(index: int, total: int, filename: str, meta: dict) -> None:
    original = normalize_phrase(str(meta.get("text", ""))) or "（空）"
    corrected = normalize_phrase(str(meta.get("corrected_text", ""))) or original
    correction_kind = str(meta.get("correction_kind") or "unknown")
    needs_manual_review = bool(meta.get("needs_manual_review"))
    correction_reason = normalize_phrase(str(meta.get("correction_reason", ""))) or "（无）"
    path = str(meta.get("path", "")).replace("\\", "/") or filename

    print(f"[{index}/{total}] {filename}")
    print(f"  path: {path}")
    print(f"  原文: {original}")
    print(f"  改后: {corrected}")
    print(f"  类型: {correction_kind}")
    print(f"  人工复核: {'是' if needs_manual_review else '否'}")
    print(f"  理由: {correction_reason}")


def print_error_log(index: int, total: int, filename: str, meta: dict, message: str) -> None:
    original = normalize_phrase(str(meta.get("text", ""))) or "（空）"
    path = str(meta.get("path", "")).replace("\\", "/") or filename

    print(f"[{index}/{total}] {filename} -> 错误", file=sys.stderr)
    print(f"  path: {path}", file=sys.stderr)
    print(f"  原文: {original}", file=sys.stderr)
    print(f"  原因: {message}", file=sys.stderr)


def print_batch_log(batch_index: int, batch_total: int, start: int, end: int, total: int) -> None:
    print(f"\n批次 {batch_index}/{batch_total} | 处理 {start}-{end}/{total}")


def print_batch_fallback_log(
    batch_index: int,
    batch_total: int,
    start: int,
    end: int,
    total: int,
    message: str,
) -> None:
    print(
        f"\n批次 {batch_index}/{batch_total} | 处理 {start}-{end}/{total} -> 批量失败，拆单条重试",
        file=sys.stderr,
    )
    print(f"  原因: {message}", file=sys.stderr)


def process_single_review_item(
    *,
    filename: str,
    meta: dict,
    offset: int,
    total: int,
    model: str,
    api_style: str,
    base_url: str,
    api_key: str,
    phrases: list[str],
    contexts: dict[str, dict[str, list[str]]],
    candidate_limit: int,
    max_retries: int,
    retry_backoff: float,
    on_progress: Callable[[], None] | None = None,
) -> None:
    single_request_item = build_request_item(
        filename=filename,
        meta=meta,
        phrases=phrases,
        contexts=contexts,
        candidate_limit=candidate_limit,
    )
    try:
        payload = call_provider_with_retries(
            api_style=api_style,
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=build_messages([single_request_item]),
            max_retries=max(1, max_retries),
            retry_backoff=max(0.0, retry_backoff),
        )
        review = parse_batch_reviews(payload, [filename])[filename]
        apply_review(meta, review, model)
        print_review_log(offset, total, filename, meta)
    except RequestFailure as single_exc:
        mark_error(meta, model, str(single_exc))
        print_error_log(offset, total, filename, meta, str(single_exc))
    except Exception as single_exc:  # noqa: BLE001
        mark_error(meta, model, f"未预期异常: {single_exc}")
        print_error_log(offset, total, filename, meta, f"未预期异常: {single_exc}")
    finally:
        if on_progress is not None:
            on_progress()


def fallback_batch_to_single_retry(
    *,
    batch: list[tuple[str, dict]],
    batch_index: int,
    batch_total: int,
    batch_start: int,
    batch_end: int,
    total: int,
    message: str,
    model: str,
    api_style: str,
    base_url: str,
    api_key: str,
    phrases: list[str],
    contexts: dict[str, dict[str, list[str]]],
    candidate_limit: int,
    max_retries: int,
    retry_backoff: float,
    on_progress: Callable[[], None] | None = None,
) -> None:
    print_batch_fallback_log(
        batch_index=batch_index,
        batch_total=batch_total,
        start=batch_start,
        end=batch_end,
        total=total,
        message=message,
    )
    for offset, (filename, meta) in enumerate(batch, start=batch_start):
        process_single_review_item(
            filename=filename,
            meta=meta,
            offset=offset,
            total=total,
            model=model,
            api_style=api_style,
            base_url=base_url,
            api_key=api_key,
            phrases=phrases,
            contexts=contexts,
            candidate_limit=candidate_limit,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            on_progress=on_progress,
        )


def prepare_review_queue(
    results: dict,
    force: bool,
    limit: int | None,
) -> tuple[list[tuple[str, dict]], bool]:
    queue: list[tuple[str, dict]] = []
    changed = False
    for filename, meta in results.items():
        if backfill_structured_review_fields(meta):
            changed = True
        text = normalize_phrase(str(meta.get("text", "")))
        if not text:
            meta["correction_status"] = "skipped"
            meta["correction_reason"] = "原始文本为空"
            changed = True
            continue
        if should_skip(meta, force):
            continue
        queue.append((filename, meta))
    if limit is not None:
        return queue[:limit], changed
    return queue, changed


def main() -> int:
    args = parse_args()
    base_url, model, api_key_env, api_style = resolve_provider_config(args)
    project_root = Path(__file__).resolve().parent.parent
    input_path = (project_root / args.input).resolve()
    manual_phrases_path = (project_root / args.manual_phrases).resolve()

    results = load_json(input_path)
    phrases = merge_phrase_candidates(results, manual_phrases_path, args.min_frequency)
    contexts = build_neighbor_context(results, max(0, args.context_window))
    queue, dirty = prepare_review_queue(results, args.force, args.limit)
    batches = chunk_queue(queue, args.batch_size)

    print(f"待纠错条目: {len(queue)}")
    print(f"候选短语总数: {len(phrases)}")
    print(f"上下文窗口: {max(0, args.context_window)}")
    print(f"批大小: {max(1, args.batch_size)}")
    print(f"提供商: {args.provider} | 模型: {model}")

    if args.dry_run:
        for filename, meta in queue[:5]:
            request_item = build_request_item(
                filename=filename,
                meta=meta,
                phrases=phrases,
                contexts=contexts,
                candidate_limit=args.candidate_limit,
            )
            print(f"\n[{filename}] {request_item['text']}")
            if request_item["previous_context"]:
                print("  前文:")
                for item in request_item["previous_context"]:
                    print(f"    - {item}")
            if request_item["next_context"]:
                print("  后文:")
                for item in request_item["next_context"]:
                    print(f"    - {item}")
            for candidate in request_item["candidates"][:8]:
                print(f"  - {candidate}")
        print("\ndry-run 完成，未调用 API。")
        return 0

    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"缺少 API key，请设置环境变量 {api_key_env}", file=sys.stderr)
        return 1

    processed_count = 0
    pending_since_save = 0

    def flush_if_needed() -> tuple[bool, int]:
        nonlocal dirty, pending_since_save
        if args.save_every > 0 and pending_since_save >= args.save_every:
            save_json(input_path, results)
            dirty = False
            pending_since_save = 0
        return dirty, pending_since_save

    def mark_progress() -> tuple[bool, int, int]:
        nonlocal dirty, processed_count, pending_since_save
        dirty = True
        processed_count += 1
        pending_since_save += 1
        flush_if_needed()
        return dirty, processed_count, pending_since_save

    for batch_index, batch in enumerate(batches, start=1):
        batch_start = processed_count + 1
        batch_end = processed_count + len(batch)
        print_batch_log(batch_index, len(batches), batch_start, batch_end, len(queue))

        batch_request_items = [
            build_request_item(
                filename=filename,
                meta=meta,
                phrases=phrases,
                contexts=contexts,
                candidate_limit=args.candidate_limit,
            )
            for filename, meta in batch
        ]
        expected_filenames = [item["filename"] for item in batch_request_items]

        try:
            payload = call_provider_with_retries(
                api_style=api_style,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=build_messages(batch_request_items),
                max_retries=max(1, args.max_retries),
                retry_backoff=max(0.0, args.retry_backoff),
            )
            reviews_by_filename = parse_batch_reviews(payload, expected_filenames)
            for offset, (filename, meta) in enumerate(batch, start=batch_start):
                apply_review(meta, reviews_by_filename[filename], model)
                print_review_log(offset, len(queue), filename, meta)
                mark_progress()
        except RequestFailure as batch_exc:
            if len(batch) == 1:
                filename, meta = batch[0]
                mark_error(meta, model, str(batch_exc))
                print_error_log(batch_start, len(queue), filename, meta, str(batch_exc))
                mark_progress()
                continue

            fallback_batch_to_single_retry(
                batch=batch,
                batch_index=batch_index,
                batch_total=len(batches),
                batch_start=batch_start,
                batch_end=batch_end,
                total=len(queue),
                message=str(batch_exc),
                model=model,
                api_style=api_style,
                base_url=base_url,
                api_key=api_key,
                phrases=phrases,
                contexts=contexts,
                candidate_limit=args.candidate_limit,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                on_progress=mark_progress,
            )
        except Exception as batch_exc:  # noqa: BLE001
            fallback_batch_to_single_retry(
                batch=batch,
                batch_index=batch_index,
                batch_total=len(batches),
                batch_start=batch_start,
                batch_end=batch_end,
                total=len(queue),
                message=f"未预期异常: {batch_exc}",
                model=model,
                api_style=api_style,
                base_url=base_url,
                api_key=api_key,
                phrases=phrases,
                contexts=contexts,
                candidate_limit=args.candidate_limit,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                on_progress=mark_progress,
            )

    if dirty:
        save_json(input_path, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
