"""
transcriber.py — Whisper 转写核心
Mac M 系列：使用 mlx-whisper（Apple Silicon 优化）
GPU 机器：使用 faster-whisper（CUDA 加速）
"""

from __future__ import annotations

import platform
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

        print(f"转写后端: {self.backend} | 模型: {config.WHISPER_MODEL}")

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
        # 计算时长
        segments = result.get("segments", [])
        duration = segments[-1]["end"] if segments else 0.0

        return {"text": text, "lang": lang, "duration": round(duration, 2)}

    def _transcribe_faster(self, audio_path: str) -> dict:
        segments, info = self._model.transcribe(
            audio_path,
            language=config.LANGUAGE,
            beam_size=config.BEAM_SIZE,
            vad_filter=config.VAD_FILTER,
        )
        segments = list(segments)
        text = "".join(s.text for s in segments).strip()
        duration = segments[-1].end if segments else 0.0

        return {
            "text": text,
            "lang": info.language,
            "duration": round(duration, 2)
        }
