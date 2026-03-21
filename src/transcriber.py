"""
transcriber.py — Whisper 转写核心
Mac M 系列：使用 mlx-whisper（Apple Silicon 优化）
GPU 机器：使用 faster-whisper（CUDA 加速）
"""

from __future__ import annotations

import platform
import opencc
import config

IS_MAC_SILICON = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)


class Transcriber:
    def __init__(self, device: str | None = None):
        self.device = device or config.DEVICE
        self._model = None
        # turbo 系列仓库不带 -mlx 后缀
        if "turbo" in config.WHISPER_MODEL:
            self._mlx_repo = f"mlx-community/whisper-{config.WHISPER_MODEL}"
        else:
            self._mlx_repo = f"mlx-community/whisper-{config.WHISPER_MODEL}-mlx"

        if IS_MAC_SILICON:
            self.backend = "mlx"
        elif self.device == "cuda":
            self.backend = "faster-whisper"
        else:
            self.backend = "faster-whisper"

        self._t2s = opencc.OpenCC("t2s")
        print(f"转写后端: {self.backend} | 模型: {config.WHISPER_MODEL}")

    def _normalize_text(self, text: str, lang: str) -> str:
        """后处理：中文繁转简，英文标题化"""
        if lang == "zh":
            return self._t2s.convert(text)
        else:
            return text.title()

    def _load(self):
        if self._model is not None:
            return

        if self.backend == "mlx":
            # Apple Silicon 专用，速度快
            import mlx_whisper
            # mlx_whisper 不需要预加载，transcribe 时传入 path_or_hf_repo
            self._model = mlx_whisper
        else:
            from faster_whisper import WhisperModel
            compute_type = config.COMPUTE_TYPE if self.device == "cuda" else "int8"
            self._model = WhisperModel(
                config.WHISPER_MODEL,
                device=self.device,
                compute_type=compute_type
            )
            print("faster-whisper 模型加载完成")

    def transcribe(self, audio_path: str) -> dict:
        """
        转写单个文件，返回：
        {
            "text": "识别文本",
            "lang": "zh",
            "duration": 2.3   # 秒
        }
        """
        self._load()

        if self.backend == "mlx":
            return self._transcribe_mlx(audio_path)
        else:
            return self._transcribe_faster(audio_path)

    def _transcribe_mlx(self, audio_path: str) -> dict:
        result = self._model.transcribe(
            audio_path,
            path_or_hf_repo=self._mlx_repo,
            language=config.LANGUAGE,
        )
        text = result.get("text", "").strip()
        lang = result.get("language", "unknown")
        text = self._normalize_text(text, lang)
        segments = result.get("segments", [])
        duration = segments[-1]["end"] if segments else 0.0
        if segments:
            no_speech_prob = sum(s["no_speech_prob"] for s in segments) / len(segments)
        else:
            no_speech_prob = 1.0

        return {"text": text, "lang": lang, "duration": round(duration, 2), "no_speech_prob": round(no_speech_prob, 4)}

    def _transcribe_faster(self, audio_path: str) -> dict:
        print(f"  [DEBUG] 开始转写: {audio_path}", flush=True)
        segments, info = self._model.transcribe(
            audio_path,
            language=config.LANGUAGE,
            beam_size=config.BEAM_SIZE,
            vad_filter=config.VAD_FILTER,
        )
        print(f"  [DEBUG] transcribe() 返回，开始消费 segments...", flush=True)
        segments = list(segments)
        print(f"  [DEBUG] segments 消费完成，共 {len(segments)} 段", flush=True)
        text = "".join(s.text for s in segments).strip()
        lang = info.language
        text = self._normalize_text(text, lang)
        duration = segments[-1].end if segments else 0.0
        if segments:
            no_speech_prob = sum(s.no_speech_prob for s in segments) / len(segments)
        else:
            no_speech_prob = 1.0

        return {
            "text": text,
            "lang": lang,
            "duration": round(duration, 2),
            "no_speech_prob": round(no_speech_prob, 4),
        }
