#!/bin/bash
set -e
echo "======================================"
echo " game-audio-transcriber 环境安装 (GPU)"
echo "======================================"

python3 --version || { echo "请先安装 Python 3.10+"; exit 1; }

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q

# 先装 CUDA 版 PyTorch（必须指定源，否则装到 CPU 版）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 再装其他依赖
pip install -r requirements-gpu.txt

if command -v nvidia-smi &> /dev/null; then
    echo "检测到 GPU："
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  [警告] 未检测到 GPU，将使用 CPU 模式"
fi

if ! command -v ffmpeg &> /dev/null; then
    apt-get install -y ffmpeg 2>/dev/null || echo "请手动安装 ffmpeg"
fi

echo ""
echo "安装完成！首次运行会自动下载 large-v3 模型（约 3GB），请确保磁盘空间充足"
echo ""
echo "使用方法："
echo "  source .venv/bin/activate"
echo "  python main.py --device cuda"
