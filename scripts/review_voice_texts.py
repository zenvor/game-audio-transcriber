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
1. 输出必须是 JSON 对象，字段固定为 corrected_text、changed、reason。
2. corrected_text 必须是适合做语音资源备注的短文本，不要扩写剧情，不要添加解释。
3. 优先依据《王者荣耀》常见播报、战场信号、术语和候选短语进行修正。
4. 如果原文已经基本正确，保留原意并尽量少改。
5. 如果无法确定，不要强行脑补，尽量保守修正。
6. 英文播报保持常见游戏播报写法；中文播报保持自然、简洁、符合游戏语境。
7. 不要无故删减信息。像 "Blue Team Rampage" 这类包含阵营信息的完整播报，除非确定有错，否则不能简化成 "Rampage!"。
8. 如果 changed=false，则 corrected_text 应与原文保持一致或只做极轻微标点修复。
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
    return meta.get("correction_status") == "reviewed"


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
  "reason": "简短说明为什么这么改"
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


def apply_review(meta: dict, review: dict, model: str) -> None:
    corrected = normalize_phrase(str(review.get("corrected_text", ""))) or normalize_phrase(str(meta.get("text", "")))
    changed = bool(review.get("changed"))
    reason = normalize_phrase(str(review.get("reason", ""))) or ("模型认为原文可保留" if not changed else "模型建议按游戏语境修正")

    meta["corrected_text"] = corrected
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
