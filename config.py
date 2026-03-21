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

# ── VAD 配置（人声检测）────────────────────────────────
VAD_THRESHOLD  = 0.5    # 0~1，越高越严格（减少误判）
VAD_MIN_SPEECH = 0.3    # 最短人声时长（秒），低于此判定为非人声

# ── 转写配置 ──────────────────────────────────────────
LANGUAGE       = None   # None = 自动检测中英文
BEAM_SIZE      = 5
VAD_FILTER     = True   # Whisper 内置二次 VAD

# ── 日志 ─────────────────────────────────────────────
LOG_INTERVAL = 50       # 每处理 N 个文件打印一次进度
