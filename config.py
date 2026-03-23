import platform

# ── 平台检测 ──────────────────────────────────────────
IS_MAC_SILICON = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)

# ── 路径 ─────────────────────────────────────────────
INPUT_DIR  = "./input"
OUTPUT_DIR = "./output"
OUTPUT_FILE = "results.json"
FAILED_FILE = "failed.json"

# ── 支持的音频格式 ────────────────────────────────────
SUPPORTED_FORMATS = (".wav",)

# ── Whisper 配置 ──────────────────────────────────────
# Mac 用 large-v3-turbo（mlx-whisper，decoder 仅 4 层，短音频快）
# GPU 机器用 large-v3（faster-whisper，精度最高，显存够用）
WHISPER_MODEL = "large-v3-turbo" if IS_MAC_SILICON else "large-v3"

# faster-whisper（GPU 机器）
DEVICE       = "cpu"                # 运行时可通过 --device cuda 覆盖
COMPUTE_TYPE = "float16"            # GPU: float16 / CPU: int8

# ── 人声/音效分流阈值 ──────────────────────────────────
NO_SPEECH_THRESHOLD = 0.6   # Whisper no_speech_prob 高于此值初步判定为纯音效
# 若 Whisper 转写出了非空文本，即使 no_speech_prob 超过上面阈值，
# 仍可能是人声（短音频填充 30s 窗口导致 nsp 偏高）；
# 只有 no_speech_prob 超过此第二阈值才最终判为音效。
NO_SPEECH_THRESHOLD_WITH_TEXT = 0.85

# 多信号补充拦截阈值（用于捕获未收录的幻觉文本）
HALLUCINATION_COMPRESSION_RATIO = 2.4   # compression_ratio >= 此值时疑似幻觉
HALLUCINATION_AVG_LOGPROB = -1.0        # avg_logprob <= 此值时疑似幻觉
HALLUCINATION_EXTREME_COMPRESSION_RATIO = 5.0  # 极端 compression_ratio，独立触发 R0

# ── 转写配置 ──────────────────────────────────────────
LANGUAGE       = None   # None = 自动检测中英文
BEAM_SIZE      = 5
VAD_FILTER     = False  # 已用 no_speech_prob 分流，无需内置 VAD

# Silero VAD 预分流开关（True 时在 Whisper 之前先用 VAD 过滤纯音效）
USE_VAD_PREFILTER = True
VAD_THRESHOLD = 0.7  # Silero VAD 语音概率阈值，默认 0.5 对游戏音效太敏感

# ── 日志 ─────────────────────────────────────────────
LOG_INTERVAL = 50       # 每处理 N 个文件打印一次进度
