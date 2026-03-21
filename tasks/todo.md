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
- [ ] 执行首个提交
- [ ] 验证提交结果

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
