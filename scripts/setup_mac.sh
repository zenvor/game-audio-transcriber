#!/bin/bash
# Mac M 系列环境安装脚本

set -e
echo "======================================"
echo " game-audio-transcriber 环境安装 (Mac)"
echo "======================================"

# 检查 Python 版本
python3 --version || { echo "请先安装 Python 3.10+"; exit 1; }

# 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 升级 pip
pip install --upgrade pip -q

# 安装依赖
echo "安装 Python 依赖..."
pip install -r requirements-mac.txt

# 检查 ffmpeg（用于音频转换）
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "安装 ffmpeg..."
    if command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "  [提示] 请手动安装 ffmpeg: brew install ffmpeg"
    fi
else
    echo "ffmpeg 已安装 ✓"
fi

echo ""
echo "======================================"
echo " 安装完成！"
echo "======================================"
echo ""
echo "使用方法："
echo "  source .venv/bin/activate"
echo "  把音频文件放入 ./input 目录"
echo "  python main.py"
echo ""
echo "调试（用小模型快速测试）："
echo "  python main.py --model medium"
