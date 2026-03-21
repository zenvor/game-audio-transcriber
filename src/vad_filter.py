"""
vad_filter.py — 用 silero-vad 判断音频是否含有人声
过滤掉纯音效、BGM，只保留需要转写的语音文件
"""

from __future__ import annotations

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

_model = None


def _load_model():
    global _model
    if _model is None:
        print("加载 VAD 模型...", end=" ", flush=True)
        _model = load_silero_vad()
        print("完成")
    return _model


def has_speech(audio_path: str, threshold: float = 0.5, min_speech_sec: float = 0.3) -> bool:
    """
    判断音频文件是否含有人声。
    threshold: VAD 置信度阈值（0~1）
    min_speech_sec: 最短人声时长（秒），低于此认为是误判
    """
    try:
        model = _load_model()

        wav = read_audio(audio_path, sampling_rate=16000)
        timestamps = get_speech_timestamps(
            wav, model,
            threshold=threshold,
            sampling_rate=16000
        )

        if not timestamps:
            return False

        # 计算总人声时长
        total_speech = sum(
            (t["end"] - t["start"]) / 16000 for t in timestamps
        )
        return total_speech >= min_speech_sec

    except Exception as e:
        # 读取失败的文件默认当作有人声（让 Whisper 去处理）
        return True


def filter_files(
    file_list: list[str],
    threshold: float = 0.5,
    min_speech_sec: float = 0.3
) -> tuple[list[str], list[str]]:
    """
    批量过滤，返回 (人声文件列表, 纯音效列表)
    """
    speech_files = []
    sfx_files = []

    total = len(file_list)
    print(f"VAD 过滤中，共 {total} 个文件...")

    for i, path in enumerate(file_list):
        if has_speech(path, threshold, min_speech_sec):
            speech_files.append(path)
        else:
            sfx_files.append(path)

        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{total} | 人声: {len(speech_files)} | 音效: {len(sfx_files)}")

    print(f"VAD 完成 → 人声 {len(speech_files)} 个，音效 {len(sfx_files)} 个")
    return speech_files, sfx_files
