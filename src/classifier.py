"""
classifier.py — 用 YAMNet 对纯音效文件进行分类标注
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import resampy

_model = None

def _load_model():
    global _model
    if _model is None:
        import tensorflow_hub as hub
        print("加载 YAMNet 模型...", end=" ", flush=True)
        _model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("完成")
    return _model

def classify(audio_path: str, top_k: int = 3) -> list[dict]:
    """
    对音频文件进行分类，返回 top_k 个最可能的类别。
    返回格式：[{"label": "Explosion", "score": 0.87}, ...]
    """
    try:
        import tensorflow as tf
        model = _load_model()

        # 读取音频，重采样到 16000Hz
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # 转单声道
        if sr != 16000:
            wav = resampy.resample(wav, sr, 16000)

        wav = wav.astype(np.float32)
        scores, embeddings, spectrogram = model(wav)
        mean_scores = scores.numpy().mean(axis=0)

        # 读取 YAMNet 类别标签
        class_map = tf.io.read_file(model.class_map_path()).numpy().decode()
        labels = [line.split(",")[2].strip()
                  for line in class_map.strip().split("\n")[1:]]

        top_indices = np.argsort(mean_scores)[::-1][:top_k]
        return [
            {"label": labels[i], "score": round(float(mean_scores[i]), 3)}
            for i in top_indices
        ]

    except Exception as e:
        return [{"label": "unknown", "score": 0.0, "error": str(e)}]


def batch_classify(file_list: list[str], top_k: int = 3) -> dict:
    """
    批量分类，返回 {文件名: [分类结果]} 映射
    """
    from pathlib import Path

    results = {}
    total = len(file_list)
    print(f"\n音效分类中，共 {total} 个文件...")

    for i, path in enumerate(file_list):
        filename = Path(path).name
        labels = classify(path, top_k)
        text = " / ".join(l["label"] for l in labels if l["label"] != "unknown")
        results[filename] = {
            "text": text,
            "type": "sfx",
            "labels": labels,
            "path": path
        }
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{total}")

    print(f"音效分类完成，共 {total} 个")
    return results
