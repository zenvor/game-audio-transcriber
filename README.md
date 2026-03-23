# game-audio-transcriber

![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

用于批量处理游戏音频资源的工具，支持语音转写、纯音效分类、语音备注纠错、文件整理与索引导出。

## 快速跑通

1. 按你的平台完成环境安装。
2. 把待处理的 `.wav` 文件放进 `input/`（支持保留原子目录结构）。
3. 运行主程序：

```bash
python main.py
```

如果是 Linux / Windows GPU 机器，通常使用：

```bash
python main.py --device cuda
```

4. 检查输出：
   `output/results.json`：有人声的音频转写结果
   `output/sfx_results.json`：纯音效的分类结果
5. 可选后处理：
   `scripts/review_voice_texts.py`：纠正语音文本
   `scripts/rename_audio_from_results.py`：按结果整理音频
   `scripts/export_audio_index.py`：导出 CSV / JSON 索引

## 关键目录

```text
input/                     # 待处理的原始 .wav 文件
output/                    # 主程序输出目录
data/honor_of_kings_phrases.txt
                           # 语音纠错用的人工短语词表
scripts/review_voice_texts.py
                           # 文本纠错
scripts/rename_audio_from_results.py
                           # 批量整理并生成重命名计划
scripts/retranscribe_and_rename.py
                           # 对已整理的音频重新识别并原地重命名
scripts/export_audio_index.py
                           # 导出索引清单
```

## 处理流程

```
所有 wav 文件
    ↓ Whisper 全量转写
    ├── no_speech_prob 低 → 有人声 → 保留转写文字 → results.json
    └── no_speech_prob 高 → 纯音效 → CLAP 分类 → sfx_results.json
```

利用 Whisper 返回的 `no_speech_prob`（无人声概率）自动区分语音和音效，无需额外的 VAD 模型。

对短音频，脚本会使用双阈值做更保守的人声判断：如果 `no_speech_prob` 略高，但 Whisper 仍转出了非空文本，仍会优先视为人声，减少短语音被误分到 `sfx_results.json` 的情况。

## 安装

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
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-mac.txt
python main.py
```

首次运行会自动下载 large-v3-turbo 模型（约 3GB）和 CLAP 模型（约 1.86GB），之后走缓存。

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
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
python main.py --device cuda
```

### Windows GPU 机器

前置条件：
- **Python 3.10 ~ 3.12**（3.13 存在依赖兼容问题，推荐 3.11）
- NVIDIA 显卡 + CUDA 驱动已装好（`nvidia-smi` 能正常输出）
- ffmpeg 已装（`winget install ffmpeg` 或从官网下载加到 PATH）

如果已装 Python 3.13，请从 python.org 额外安装 3.11，然后用 `py -3.11` 创建虚拟环境。

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
python main.py --device cuda
```

> **踩坑提醒：**
> - **Python 版本**：必须用 3.10 ~ 3.12。Python 3.13 下 `ctranslate2` 要求 `numpy<2`，而 `numpy 1.x` 在 3.13 上没有预编译轮子，会报编译器找不到的错误。
> - **pip 升级**：必须在 `pip install -r` 之前先升级 pip。旧版 pip 在 Windows 上用 GBK 编码读取文件，遇到 UTF-8 中文注释会报 `UnicodeDecodeError`。
> - **多版本共存**：Windows 上可以同时装多个 Python 版本，用 `py -3.11` 指定版本，互不影响。

首次运行会自动下载 large-v3 模型（约 3GB）和 CLAP 模型（约 1.86GB），之后走缓存。

## 用法

### 主程序

```
python main.py [OPTIONS]

  --input DIR       输入目录（默认 ./input）
  --output DIR      输出目录（默认 ./output）
  --device STR      运行设备：cuda / cpu（默认自动检测）
  --model NAME      覆盖 Whisper 模型（默认 large-v3-turbo / large-v3）
  --sfx-only        仅重新分类音效，跳过 Whisper 转写
  --recheck-sfx     重新检查已有 sfx_results.json，把误分流的人声迁回 results.json
```

如果历史结果里已有一批短语音被误分到 `output/sfx_results.json`，可以单独执行：

```bash
python main.py --recheck-sfx
```

这个模式会重新转写 `sfx_results.json` 里的文件，并用当前双阈值规则判断是否应迁回 `results.json`；它不会重新跑 CLAP 分类，也不会扫描 `input/` 中的新文件。

### 语音纠错

```
python3 scripts/review_voice_texts.py [OPTIONS]

  --input FILE      语音结果文件（默认 output/results.json）
  --provider STR    接口提供商预设：gemini / openai（默认 gemini）
  --model NAME      模型名称，默认使用 provider 预设
  --force           即使已有结果也重新处理
  --limit N         只处理前 N 条，便于验证
  --context-window N 每侧附带的相邻上下文条数（默认 2）
  --batch-size N    每批请求处理的条目数（默认 5；设为 1 表示单条模式）
  --dry-run         只预览候选短语、相邻上下文和待处理样本，不调用 API

高级选项（一般不需要改）：--base-url, --api-key-env, --manual-phrases,
  --min-frequency, --candidate-limit, --save-every, --max-retries, --retry-backoff
```

### 音频整理重命名

```
python3 scripts/rename_audio_from_results.py [OPTIONS]

  --results FILE      语音结果文件路径（默认 output/results.json）
  --sfx-results FILE  音效结果文件路径（默认 output/sfx_results.json）
  --plan-out FILE     重命名计划输出路径（默认 output/rename_plan.json）
  --target-root DIR   分类重命名输出根目录（默认 output/renamed_audio）
  --apply             实际执行复制重命名；默认只预览并导出计划
```

### 重新识别并重命名

对已整理到 `output/renamed_audio/` 下的音频文件重新跑 Whisper 转录，并根据新结果原地重命名。适用于部分文件转录不准或被错误分类的情况。

```
python3 scripts/retranscribe_and_rename.py <路径...> [OPTIONS]

  <路径...>             音频文件或目录（目录会递归扫描 .wav），支持多个
  --device STR          运行设备：cuda / cpu（默认自动检测）
  --model NAME          覆盖 Whisper 模型
  --dry-run             只预览，不实际重命名
```

先 dry-run 预览，确认无误后去掉 `--dry-run` 执行：

```bash
# 预览单个文件
python3 scripts/retranscribe_and_rename.py output/renamed_audio/sfx/某文件.wav --dry-run

# 预览整个目录
python3 scripts/retranscribe_and_rename.py output/renamed_audio/sfx/ --dry-run

# 确认后实际执行
python3 scripts/retranscribe_and_rename.py output/renamed_audio/sfx/
```

### 导出音频索引

```
python3 scripts/export_audio_index.py [OPTIONS]

  --plan FILE      重命名计划文件路径（默认 output/rename_plan.json）
  --csv-out FILE   CSV 索引输出路径（默认 output/audio_index.csv）
  --json-out FILE  JSON 索引输出路径（默认 output/audio_index.json）
  --status STR     只导出指定状态；传 all 导出全部（默认 planned）
```

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

## 语音纠错详解

这里做的是"文本纠错"，不是重新听音频转写。上游文本来自 `Whisper large-v3` 转写，常见问题通常是发音相近导致的错别字、近音词、同音词、英文单词/短语误识别，以及轻微标点偏差，而不是整句被完全改写。

脚本内置 `gemini` 和 `openai` 两个 provider 预设，默认使用 `gemini`：

- `gemini`：使用 Gemini 原生 `generateContent` 接口
- `openai`：使用 OpenAI Chat Completions 接口
- 请求失败时自动有限次重试，避免瞬时网络抖动或限流导致大量条目直接标记为错误
- 模型会优先结合《王者荣耀》语境、同目录相邻条目和文本本身是否读起来合理来判断，不要求必须命中候选短语
- 默认会把待纠错条目按 5 条一批发给模型；如果整批返回非 JSON、缺项、条数不符或文件名对不上，会自动拆回单条重试

先做 dry-run，查看候选短语和待处理样本：

```bash
python3 scripts/review_voice_texts.py --dry-run --limit 5
```

正式调用前先设置对应的 API key：

```bash
# Gemini
export GEMINI_API_KEY="你的 Gemini API Key"

# OpenAI
export OPENAI_API_KEY="你的 OpenAI API Key"
```

然后执行纠错：

```bash
# Gemini（默认）
python3 scripts/review_voice_texts.py

# OpenAI
python3 scripts/review_voice_texts.py --provider openai
```

脚本会自动：

- 读取 `output/results.json`
- 合并 `data/honor_of_kings_phrases.txt` 和现有结果中的高频短语
- 为每条记录补充同目录优先的前后文；不足时再回退到结果文件中的相邻条目
- 把《王者荣耀》语境、相邻上下文、候选短语和当前文本按批次一起发给所选模型
- 根据语句是否自然、是否符合游戏固定播报来判断错别字、同音字和近音词误识别
- 本地护栏会拦截误删阵营前缀、核心播报片段的缩句式修正，但允许删除重复噪声，如 `Defeat defeat → Defeat`
- 整批返回结构异常时自动拆回单条重试；`--batch-size 1` 可显式关闭批量模式
- 将建议写回到 `corrected_text`

脚本不会覆盖原始 `text`，而是新增以下字段：

- `corrected_text`：纠错后的文本
- `correction_status`：纠错状态，例如 `reviewed` / `error`
- `correction_model`：本次纠错使用的模型名
- `correction_changed`：是否实际发生改动（以最终文本对比为准，不依赖模型返回）
- `correction_kind`：纠错类型（见下表）
- `needs_manual_review`：是否需要人工复核
- `correction_reason`：纠错理由（中文）

**纠错类型说明（`correction_kind`）：**

| 值 | 含义 |
|----|------|
| `asr_error` | 识别内容被改正，如 `Hector Kill! → Hexakill!` |
| `punctuation` | 只补标点或轻微格式修复 |
| `normalization` | 写法统一，不属于明显错听 |
| `uncertain` | 模型认为需要人工复核 |
| `no_change` | 原文保留 |

示例：

```json
{
  "01_8772748.wav": {
    "text": "Hector Kill!",
    "corrected_text": "Hexakill!",
    "correction_changed": true,
    "correction_kind": "asr_error",
    "needs_manual_review": false,
    "correction_reason": "ASR 识别有误，已将“Hector Kill!”修正为更符合游戏语境的“Hexakill!”。"
  }
}
```

## 音频整理详解

当你已经在 `output/results.json` 和 `output/sfx_results.json` 里整理好文本后，可以用脚本批量整理原始音频。

默认先做 dry-run，只生成计划，不实际复制：

```bash
python3 scripts/rename_audio_from_results.py
```

确认计划无误后，再实际执行：

```bash
python3 scripts/rename_audio_from_results.py --apply
```

脚本会：

- 对语音结果，当 `correction_changed` 为 `true` 时使用 `corrected_text`，否则使用原始 `text`
- 清洗非法文件名字符
- 将文件复制到 `output/renamed_audio/` 下，按 `speech/` 和 `sfx/` 两大类分目录
- 保留原子目录结构，例如 `System_Voice`、`UI`
- 生成目标文件名：`标注文本__原始编号.wav`
- 输出计划到 `output/rename_plan.json`

示例：

```
input/System_Voice/Hector_Kill__01_8772748.wav
→
output/renamed_audio/speech/System_Voice/Hexakill__01_8772748.wav
```

整理完成后，导出 CSV / JSON 索引，方便人工校对和资产交付：

```bash
python3 scripts/export_audio_index.py
```

默认输出 `output/audio_index.csv` 和 `output/audio_index.json`，包含原始文本、校正文、实际生效文本、文本来源、源路径和目标路径。
