# 项目初始化任务

## 目标

为当前项目完成适合首次提交到 Git 的基础初始化，保持最小改动，不影响现有代码运行。

## 待办

- [x] 检查当前目录结构与 Git 状态
- [x] 初始化 Git 仓库
- [x] 添加适合本项目的 `.gitignore`
- [x] 补充仓库基础文件（`LICENSE`、`.editorconfig`）
- [x] 为 `input/` 和 `output/` 添加占位文件，保留目录结构
- [x] 验证 `git status` 输出符合预期
- [x] 执行首个提交
- [x] 验证提交结果

## 验收标准

- 项目根目录存在 `.git/`
- Git 仅追踪源码、脚本、文档和必要占位文件
- 不追踪虚拟环境、Python 缓存、IDE 文件、运行输出、输入音频
- `git status --short` 输出清晰，可直接用于首个提交前检查

## Review

- 已执行 `git init`，仓库初始化完成
- 已添加根目录 `.gitignore`，忽略 Python 缓存、虚拟环境、`.claude/`、运行日志、本地环境文件
- 已添加标准 MIT `LICENSE`，版权人为 `zenvor`
- 已添加 `.editorconfig`，统一基础文本格式
- 已将 `input/`、`output/` 设计为仅保留 `.gitkeep`，避免原始音频和运行结果进入版本库
- 已验证 `git status --short --ignored`：
  - 待跟踪项为源码、脚本、文档、任务文件和目录占位文件
  - 已忽略项包含 `.claude/`、`output/results.json`、`output/results copy.json`
- 已创建首个提交：`53d726a chore: initialize repository`
- 提交后再次验证：
  - 工作区无待提交文件
  - 仅剩已忽略文件：`.claude/`、`output/results.json`、`output/results copy.json`

## 音频重命名任务

### 目标

根据 `output/results.json` 与 `output/sfx_results.json` 中已整理好的 `text` 字段，按语音和音效两大类整理音频，并保留原子目录。

### 待办

- [x] 检查两份结果文件的结构、路径字段和重名情况
- [x] 实现独立的批量重命名脚本
- [x] 默认支持 dry-run，并导出重命名计划
- [x] 验证脚本在当前数据集上的输出是否合理

### 重命名规则

- 读取 `results.json` 和 `sfx_results.json` 中每条记录的 `path` 与 `text`
- `text` 作为主文件名来源，但必须做非法字符清洗
- 默认保留原始文件编号作为后缀，避免大量同名音频冲突
- 输出到 `renamed_audio/speech/` 和 `renamed_audio/sfx/`
- 在分类目录下保留原始子目录结构
- 默认不立即执行，只有显式传入执行参数才真正复制整理

### Review

- 已新增脚本 `scripts/rename_audio_from_results.py`
- 脚本读取 `results.json` 与 `sfx_results.json` 中的 `text`、`path`
- 脚本把文件整理到 `renamed_audio/speech/` 与 `renamed_audio/sfx/`，并保留原子目录
- 文件名清洗后按 `标注文本__原始编号.wav` 生成目标名，避免重名覆盖
- 已兼容“原文件已被提前重命名”的场景，会自动回查 `__原始编号.wav`
- 已用当前数据集完成 dry-run：
  - `planned: 926`
  - `speech: 327`
  - `sfx: 599`
  - 未发现 `conflict_target_exists` 或 `conflict_duplicate_target`
- 已生成计划文件 `output/rename_plan.json`
- 已在 README 增加使用说明，默认先预览，确认后再 `--apply`

## 音频索引任务

### 目标

基于当前的 `output/rename_plan.json` 导出可检索的音频索引清单，便于后续查找、校对和交付。

### 待办

- [x] 检查 `rename_plan.json` 字段是否足够导出索引
- [x] 实现索引导出脚本
- [x] 导出 CSV 和 JSON 两种格式
- [x] 验证导出数量与字段

### 输出要求

- CSV 适合表格查看
- JSON 适合脚本消费
- 至少包含：`category`、`original_name`、`text`、`source_path`、`target_path`、`status`

### Review

- 已新增脚本 `scripts/export_audio_index.py`
- 基于 `output/rename_plan.json` 导出索引，避免重复扫描音频目录
- 已生成：
  - `output/audio_index.csv`
  - `output/audio_index.json`
- 已验证导出数量：
  - CSV: 926 条
  - JSON: 926 条
- 已验证字段包含：
  - `category`
  - `status`
  - `original_name`
  - `text`
  - `sanitized_text`
  - `requested_source_path`
  - `source_path`
  - `target_dir`
  - `target_path`
  - `source_json`
