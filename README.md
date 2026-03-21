# game-audio-transcriber

批量将游戏音频包（语音/音效）转写为文字。

## 项目结构

```
game-audio-transcriber/
├── input/              # 放原始音频文件（.wav）
├── output/             # 转写结果输出
├── src/
│   ├── __init__.py
│   ├── classifier.py   # YAMNet 音效分类
│   ├── transcriber.py  # Whisper 转写核心
│   └── pipeline.py     # 完整流水线入口
├── scripts/
│   ├── setup_mac.sh    # Mac M 系列环境安装
│   └── setup_gpu.sh    # Linux GPU 机器环境安装
├── config.py           # 配置文件
├── main.py             # 主入口
├── requirements-mac.txt
└── requirements-gpu.txt
```

## 处理流程

```
所有 wav 文件
    ↓ Whisper 全量转写
    ├── no_speech_prob 低 → 有人声 → 保留转写文字 → results.json
    └── no_speech_prob 高 → 纯音效 → YAMNet 分类 → sfx_results.json
```

利用 Whisper 返回的 `no_speech_prob`（无人声概率）自动区分语音和音效，无需额外的 VAD 模型。

## 模型说明

- Mac M 系列：large-v3-turbo 模型（mlx-whisper，decoder 仅 4 层，短音频快）
- GPU 机器：large-v3 模型（faster-whisper，精度最高）

可通过 `--model` 参数覆盖，例如 `python main.py --model medium`。

## 快速开始

### Mac M 系列（Apple Silicon）

前置条件：
- Python 3.9+
- ffmpeg 已装（`brew install ffmpeg`）

**方式一：一键脚本**
```bash
bash scripts/setup_mac.sh
```

**方式二：手动安装**
```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements-mac.txt

# 运行
python main.py
```

首次运行会自动下载 large-v3-turbo 模型（约 3GB），之后走缓存。

常用参数：
```bash
python main.py --input ./你的音频目录    # 指定输入目录（默认 ./input）
python main.py --output ./你的输出目录   # 指定输出目录（默认 ./output）
python main.py --model medium           # 用较小模型（调试用）
```

### Linux GPU 机器
```bash
bash scripts/setup_gpu.sh
python main.py --input ./input --output ./output --device cuda
```

### Windows GPU 机器

前置条件：
- Python 3.10+
- NVIDIA 显卡 + CUDA 驱动已装好（`nvidia-smi` 能正常输出）
- ffmpeg 已装（`winget install ffmpeg` 或从官网下载加到 PATH）

```powershell
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 升级 pip
pip install --upgrade pip

# 安装 CUDA 版 PyTorch（faster-whisper 的 CTranslate2 依赖其 CUDA 运行时库）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install -r requirements-gpu.txt

# 运行
python main.py --device cuda
```

首次运行会自动下载 large-v3 模型（约 3GB），之后走缓存。

常用参数：
```powershell
python main.py --device cuda --input .\你的音频目录    # 指定输入目录（默认 .\input）
python main.py --device cuda --output .\你的输出目录   # 指定输出目录（默认 .\output）
python main.py --device cuda --model medium           # 用较小模型（调试用）
```

## 输出格式

```
output/
├── results.json      # 人声文件 → 转写文字
└── sfx_results.json  # 纯音效文件 → YAMNet 分类标签
```

`output/results.json`：
```json
{
  "hero_win.wav": {
    "text": "胜利属于我们！",
    "lang": "zh",
    "duration": 2.3,
    "no_speech_prob": 0.02,
    "path": "./input/hero_win.wav"
  }
}
```

`output/sfx_results.json`：
```json
{
  "attack_01.wav": {
    "type": "sfx",
    "labels": [
      {"label": "Sword", "score": 0.82},
      {"label": "Clang", "score": 0.11},
      {"label": "Battle cry", "score": 0.04}
    ],
    "path": "./input/attack_01.wav"
  }
}
```
