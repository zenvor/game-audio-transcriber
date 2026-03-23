"""
vad.py — Silero VAD 封装
用于在 Whisper 转写之前快速判断音频文件是否包含人声。
"""

from __future__ import annotations


class SileroVAD:
    def __init__(self) -> None:
        self._model = None
        self._read_audio = None
        self._get_speech_timestamps = None
        self._unavailable = False  # 首次加载失败后置 True，不再重试

    def _load(self) -> None:
        if self._model is not None or self._unavailable:
            return
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
        self._model = load_silero_vad()
        self._read_audio = read_audio
        self._get_speech_timestamps = get_speech_timestamps
        print("Silero VAD 模型加载完成")

    def has_speech(self, audio_path: str) -> bool:
        """
        判断音频文件中是否包含人声。
        出错时保守返回 True，交给 Whisper 进一步判断，避免漏掉人声。
        模型加载失败后缓存失败状态，不再对后续文件重复重试。
        """
        if self._unavailable:
            return True
        try:
            self._load()
            from config import VAD_THRESHOLD
            wav = self._read_audio(audio_path)  # 内部自动重采样到 16kHz
            timestamps = self._get_speech_timestamps(
                wav, self._model, threshold=VAD_THRESHOLD,
            )
            return len(timestamps) > 0
        except Exception as exc:
            if self._model is None:
                # 加载阶段失败，后续无需重试
                self._unavailable = True
                print(f"  [VAD] 模型不可用（{exc}），后续文件跳过 VAD 直接转写")
            else:
                # 单文件处理失败，不影响后续
                print(f"  [VAD 警告] {audio_path}: {exc}，保守归为待转写")
            return True
