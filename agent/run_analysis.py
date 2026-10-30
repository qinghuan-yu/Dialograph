"""
聊天记录分析 Agent - 预处理与采样提示词入口
运行此脚本生成预处理材料与采样分析提示词。
"""

import sys
from pathlib import Path


def main():
    from preprocess_cli import run_preprocessing_cli
    return run_preprocessing_cli(Path(__file__).parent.parent,
                                 sys.argv[1] if len(sys.argv) > 1 else None,
                                 include_preview_prompt=True)


if __name__ == "__main__":
    main()
