# game-audio-transcriber

用于批量处理游戏音频资源的工具，支持语音转写、纯音效分类、语音备注纠错、文件整理与索引导出。

## 项目结构

```
game-audio-transcriber/
├── data/
│   └── honor_of_kings_phrases.txt   # 王者荣耀短语词表（用于文本纠错）
├── input/              # 放原始音频文件（.wav）
├── output/             # 转写结果输出
├── src/
│   ├── __init__.py
│   ├── classifier.py   # CLAP 音效零样本分类
│   ├── transcriber.py  # Whisper 转写核心
│   └── pipeline.py     # 完整流水线入口
├── scripts/
│   ├── export_audio_index.py    # 导出音频索引
│   ├── rename_audio_from_results.py   # 按结果整理音频
│   ├── review_voice_texts.py    # 用 Gemini / OpenAI 纠正语音文本
│   ├── setup_mac.sh             # Mac M 系列环境安装
│   └── setup_gpu.sh             # Linux GPU 机器环境安装
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
- 内置数百个王者荣耀音效分类标签（战斗、技能、UI、环境、角色动作等）
- 首次运行自动下载模型权重，无需手动下载
- 标签可在 `src/classifier.py` 中的 `LABELS` 列表自由修改

## 快速开始

### Mac M 系列（Apple Silicon）

前置条件：
- Python 3.10+（推荐 3.11，torch/CLAP 兼容性最佳）
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

首次运行会自动下载 large-v3-turbo 模型（约 3GB）和 CLAP 模型（约 1.86GB），之后走缓存。

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
    "text": "近战武器攻击 / 暴击 / 受击",
    "labels": [
      {"label": "近战武器攻击", "score": 0.82},
      {"label": "暴击", "score": 0.11},
      {"label": "受击", "score": 0.04}
    ],
    "path": "./input/attack_01.wav"
  }
}
```

## 用 OpenAI / Gemini 校正语音文本

如果 `output/results.json` 里的语音备注存在错听，可以用 OpenAI 或 Gemini 做二次纠错。这里做的是“文本纠错”，不是重新听音频转写。该脚本不会覆盖原始 `text`，而是新增 `corrected_text`、`correction_status`、`correction_reason`、`correction_model`。

先做 dry-run，查看候选短语和待处理样本：

```bash
python3 scripts/review_voice_texts.py --dry-run --limit 5
```

脚本内置 `gemini` 和 `openai` 两个 provider 预设，默认使用 `gemini`。其中：

- `gemini` 使用 Gemini 原生 `generateContent` 接口
- `openai` 使用 OpenAI Chat Completions 接口
- 请求失败时会自动做有限次重试，避免瞬时网络抖动或限流导致大量条目直接标记为错误
- 结果会按批次写回 JSON，而不是每处理一条就完整重写一次文件

正式调用前先设置对应的 API key。

Gemini：

```bash
export GEMINI_API_KEY="你的 Gemini API Key"
```

OpenAI：

```bash
export OPENAI_API_KEY="你的 OpenAI API Key"
```

然后执行纠错。

Gemini（默认）：

```bash
python3 scripts/review_voice_texts.py
```

也可以显式指定：

```bash
python3 scripts/review_voice_texts.py --provider gemini
```

OpenAI：

```bash
python3 scripts/review_voice_texts.py --provider openai
```

常用参数：

```bash
python3 scripts/review_voice_texts.py --limit 20
python3 scripts/review_voice_texts.py --force
python3 scripts/review_voice_texts.py --provider gemini --model gemini-3-flash-preview
python3 scripts/review_voice_texts.py --provider openai --model gpt-4.1-mini
```

脚本会自动：

- 读取 `output/results.json`
- 合并 `data/honor_of_kings_phrases.txt` 和现有结果中的高频短语
- 把王者荣耀语境和候选短语一起发给所选模型
- 将建议写回到 `corrected_text`

如果你已经手动确认某些纠错结果不可靠，可以直接删除这四个新增字段，回退到原始 `text`。

## 批量分类整理音频

当你已经在 `output/results.json` 和 `output/sfx_results.json` 里整理好文本后，可以用脚本批量整理原始音频。

默认先做 dry-run，只生成计划，不直接改名：

```bash
python3 scripts/rename_audio_from_results.py
```

脚本会：

- 读取两份结果文件中的 `path` 和文本字段
- 对语音结果优先使用 `corrected_text`，没有时回退到 `text`
- 按显式的 `speech` / `sfx` 类别处理两份结果文件，不依赖默认文件名推断类型
- 清洗非法文件名字符
- 将文件复制到 `renamed_audio/` 下
- 按两大类分目录：`speech/` 和 `sfx/`
- 保留原子目录结构，例如 `System_Voice`、`UI`
- 生成目标文件名：`标注文本__原始编号.wav`
- 输出计划到 `output/rename_plan.json`

确认计划无误后，再实际执行重命名：

```bash
python3 scripts/rename_audio_from_results.py --apply
```

示例：

```text
input/System_Voice/Hector_Kill__01_8772748.wav
→
renamed_audio/speech/System_Voice/Hector_Kill__01_8772748.wav
```

## 导出音频索引

整理计划生成后，可以导出 CSV / JSON 索引，方便人工校对和资产交付：

```bash
python3 scripts/export_audio_index.py
```

默认输出：

```text
output/audio_index.csv
output/audio_index.json
```

索引会包含：

- 原始文本 `text`
- 校正文 `corrected_text`
- 实际生效文本 `effective_text`
- 文本来源 `text_source`
- 源路径和目标路径
