# game-audio-transcriber

批量将游戏音频包（语音/音效）转写为文字。

## 项目结构

```
game-audio-transcriber/
├── input/              # 放原始音频文件（.wav）
├── output/             # 转写结果输出
├── src/
│   ├── __init__.py
│   ├── classifier.py   # CLAP 音效零样本分类
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
    └── no_speech_prob 高 → 纯音效 → CLAP 分类 → sfx_results.json
```

利用 Whisper 返回的 `no_speech_prob`（无人声概率）自动区分语音和音效，无需额外的 VAD 模型。

## 模型说明

**语音转写：**
- Mac M 系列：large-v3-turbo 模型（mlx-whisper，decoder 仅 4 层，短音频快）
- GPU 机器：large-v3 模型（faster-whisper，精度最高）

可通过 `--model` 参数覆盖，例如 `python main.py --model medium`。

**音效分类：**
- CLAP（LAION，零样本音频分类，支持自定义标签）
- 内置 47 个王者荣耀音效分类标签（战斗、技能、播报、地图、UI、环境、角色动作）
- 首次运行自动下载模型权重，无需手动下载
- 标签可在 `src/classifier.py` 中的 `LABELS` 列表自由修改

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

首次运行会自动下载 large-v3-turbo 模型（约 3GB）和 CLAP 模型，之后走缓存。

常用参数：
```bash
python main.py --input ./你的音频目录    # 指定输入目录（默认 ./input）
python main.py --output ./你的输出目录   # 指定输出目录（默认 ./output）
python main.py --model medium           # 用较小模型（调试用）
python main.py --sfx-only              # 仅重新分类音效（跳过 Whisper 转写）
```

### Linux GPU 机器

前置条件：
- Python 3.10 ~ 3.12（推荐 3.11）
- NVIDIA 显卡 + CUDA 驱动已装好（`nvidia-smi` 能正常输出）
- ffmpeg 已装

**方式一：一键脚本**
```bash
bash scripts/setup_gpu.sh
python main.py --device cuda
```

**方式二：手动安装**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 先装 CUDA 版 PyTorch（faster-whisper 和 CLAP 需要）
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

# 再装其他依赖
pip install -r requirements-gpu.txt

python main.py --device cuda
```

### Windows GPU 机器

前置条件：
- **Python 3.10 ~ 3.12**（3.13 存在依赖兼容问题，推荐 3.11）
- NVIDIA 显卡 + CUDA 驱动已装好（`nvidia-smi` 能正常输出）
- ffmpeg 已装（`winget install ffmpeg` 或从官网下载加到 PATH）

如果已装 Python 3.13，请从 https://www.python.org/downloads/ 额外安装 3.11，然后用 `py -3.11` 创建虚拟环境。

```powershell
# 创建虚拟环境（指定 Python 3.11）
py -3.11 -m venv .venv
.venv\Scripts\activate

# 升级 pip（必须先升级，旧版 pip 读取含中文注释的 requirements 文件会报 GBK 编码错误）
python.exe -m pip install --upgrade pip

# 安装 CUDA 版 PyTorch（faster-whisper 和 CLAP 需要）
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install -r requirements-gpu.txt

# 运行
python main.py --device cuda
```

> **踩坑提醒：**
> - **Python 版本**：必须用 3.10 ~ 3.12。Python 3.13 下 `ctranslate2`（faster-whisper 的依赖）要求 `numpy<2`，而 `numpy 1.x` 在 3.13 上没有预编译轮子，会报编译器找不到的错误。
> - **pip 升级**：必须在 `pip install -r` 之前先升级 pip。旧版 pip（如 24.0）在 Windows 上用 GBK 编码读取文件，遇到 UTF-8 中文注释会报 `UnicodeDecodeError`。
> - **多版本共存**：Windows 上可以同时装多个 Python 版本，用 `py -3.11` 指定版本，互不影响。

首次运行会自动下载 large-v3 模型（约 3GB）和 CLAP 模型（约 1.86GB），之后走缓存。

常用参数：
```powershell
python main.py --device cuda --input .\你的音频目录    # 指定输入目录（默认 .\input）
python main.py --device cuda --output .\你的输出目录   # 指定输出目录（默认 .\output）
python main.py --device cuda --model medium           # 用较小模型（调试用）
python main.py --sfx-only                             # 仅重新分类音效（跳过 Whisper 转写）
```

## 输出格式

```
output/
├── results.json      # 人声文件 → 转写文字
└── sfx_results.json  # 纯音效文件 → CLAP 分类标签
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
      {"label": "近战武器攻击", "score": 0.82},
      {"label": "暴击", "score": 0.11},
      {"label": "受击", "score": 0.04}
    ],
    "path": "./input/attack_01.wav"
  }
}
```
