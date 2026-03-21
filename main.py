"""
main.py — 入口
用法：
  python main.py                          # 用 config.py 默认配置
  python main.py --input ./音频包         # 指定输入目录
  python main.py --device cuda            # GPU 模式（迁移到 GPU 机器后）
  python main.py --model medium           # 用较小模型（调试用）
"""

import argparse
import config
from src.pipeline import run, run_sfx_only


def parse_args():
    parser = argparse.ArgumentParser(description="游戏音频批量转写工具")
    parser.add_argument("--input",    default=config.INPUT_DIR,  help="输入目录")
    parser.add_argument("--output",   default=config.OUTPUT_DIR, help="输出目录")
    parser.add_argument("--device",   default=None,              help="cuda / cpu（默认自动）")
    parser.add_argument("--model",    default=None,              help="覆盖 config 中的模型")
    parser.add_argument("--sfx-only", action="store_true",       help="仅重新分类音效文件（跳过 Whisper 转写）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model:
        config.WHISPER_MODEL = args.model

    if args.sfx_only:
        run_sfx_only(output_dir=args.output)
    else:
        run(
            input_dir=args.input,
            output_dir=args.output,
            device=args.device,
        )
