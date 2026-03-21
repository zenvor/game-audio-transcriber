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
NO_SPEECH_THRESHOLD = 0.6  # Whisper no_speech_prob 高于此值判定为纯音效

# ── 转写配置 ──────────────────────────────────────────
LANGUAGE       = None   # None = 自动检测中英文
BEAM_SIZE      = 5
VAD_FILTER     = False  # 已用 no_speech_prob 分流，无需内置 VAD

# ── 日志 ─────────────────────────────────────────────
LOG_INTERVAL = 50       # 每处理 N 个文件打印一次进度
