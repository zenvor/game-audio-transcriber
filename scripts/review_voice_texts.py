#!/usr/bin/env python3
"""使用兼容 OpenAI 或 Gemini 接口的模型对语音识别文本做上下文纠错。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from urllib import error, request


SYSTEM_PROMPT = """你是游戏本地化与语音资产整理助手。

现在需要你校正《王者荣耀》语音备注文本。输入文本来自 ASR 语音识别，可能出现错听、近音字、同音词、标点缺失、英文播报拼写错误等问题。

规则：
1. 输出必须是 JSON 对象，字段固定为 corrected_text、changed、reason、kind、needs_manual_review。
2. corrected_text 必须是适合做语音资源备注的短文本，不要扩写剧情，不要添加解释。
3. 优先依据《王者荣耀》常见播报、战场信号、术语和候选短语进行修正。
4. 如果原文已经基本正确，保留原意并尽量少改。
5. 如果无法确定，不要强行脑补，尽量保守修正。
6. 英文播报保持常见游戏播报写法；中文播报保持自然、简洁、符合游戏语境。
7. 不要无故删减信息。像 "Blue Team Rampage" 这类包含阵营信息的完整播报，除非确定有错，否则不能简化成 "Rampage!"。
8. 如果 changed=false，则 corrected_text 应与原文保持一致或只做极轻微标点修复。
9. kind 只能取以下枚举之一：asr_error、punctuation、normalization、uncertain、no_change。
10. needs_manual_review 只在你无法高置信判断，或认为仍需人工过目时设为 true。
11. reason 必须使用简体中文，禁止输出英文解释。
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
    if selected:
        return selected
    return phrases[:limit]


def should_skip(meta: dict, force: bool) -> bool:
    if force:
        return False
    return (
        meta.get("correction_status") == "reviewed"
        and "correction_changed" in meta
        and "correction_kind" in meta
        and "needs_manual_review" in meta
    )


def build_messages(text: str, lang: str, path: str, candidates: list[str]) -> list[dict]:
    candidate_block = "\n".join(f"- {item}" for item in candidates) if candidates else "- 无"
    user_prompt = f"""请校正下面这条《王者荣耀》语音备注文本。

语言: {lang or 'unknown'}
文件路径: {path}
当前识别文本: {text}

候选短语:
{candidate_block}

请输出 JSON：
{{
  "corrected_text": "校正后的文本",
  "changed": true,
  "reason": "使用简体中文简短说明为什么这么改",
  "kind": "asr_error",
  "needs_manual_review": false
}}
"""
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
    changed = resolve_correction_changed(original, corrected)
    requested_kind = normalize_correction_kind(review.get("kind"))
    if not changed:
        correction_kind = "no_change"
    elif requested_kind and requested_kind != "no_change":
        correction_kind = requested_kind
    else:
        correction_kind = infer_correction_kind(
            original=original,
            corrected=corrected,
            changed=changed,
        )
    needs_manual_review = normalize_manual_review(
        review.get("needs_manual_review"),
        correction_kind,
    )
    raw_reason = normalize_phrase(str(review.get("reason", "")))
    reason = raw_reason if contains_cjk(raw_reason) else fallback_reason_in_chinese(
        original=original,
        corrected=corrected,
        changed=changed,
        correction_kind=correction_kind,
        needs_manual_review=needs_manual_review,
    )

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
    queue, dirty = prepare_review_queue(results, args.force, args.limit)

    print(f"待纠错条目: {len(queue)}")
    print(f"候选短语总数: {len(phrases)}")
    print(f"提供商: {args.provider} | 模型: {model}")

    if args.dry_run:
        for filename, meta in queue[:5]:
            text = normalize_phrase(str(meta.get('text', '')))
            candidates = select_candidate_phrases(text, phrases, args.candidate_limit)
            print(f"\n[{filename}] {text}")
            for candidate in candidates[:8]:
                print(f"  - {candidate}")
        print("\ndry-run 完成，未调用 API。")
        return 0

    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"缺少 API key，请设置环境变量 {api_key_env}", file=sys.stderr)
        return 1

    for index, (filename, meta) in enumerate(queue, start=1):
        text = normalize_phrase(str(meta.get("text", "")))
        lang = str(meta.get("lang", ""))
        path = str(meta.get("path", ""))
        candidates = select_candidate_phrases(text, phrases, args.candidate_limit)
        try:
            review = call_provider_with_retries(
                api_style=api_style,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=build_messages(text=text, lang=lang, path=path, candidates=candidates),
                max_retries=max(1, args.max_retries),
                retry_backoff=max(0.0, args.retry_backoff),
            )
            apply_review(meta, review, model)
            print(f"[{index}/{len(queue)}] {filename} -> {meta['corrected_text']}")
        except RequestFailure as exc:
            mark_error(meta, model, str(exc))
            print(f"[{index}/{len(queue)}] {filename} -> 错误: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            mark_error(meta, model, f"未预期异常: {exc}")
            print(f"[{index}/{len(queue)}] {filename} -> 未预期异常: {exc}", file=sys.stderr)

        dirty = True
        if dirty and args.save_every > 0 and index % args.save_every == 0:
            save_json(input_path, results)
            dirty = False

    if dirty:
        save_json(input_path, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
