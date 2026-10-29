"""两个预处理命令共用的文件选择与结果展示。"""
from __future__ import annotations

from analysis_artifacts import build_artifacts, select_chat_files


def run_preprocessing_cli(workspace, selection=None, *, include_preview_prompt=False):
    files = select_chat_files(workspace / 'talks', [], None)
    if not files:
        print('未找到支持的聊天记录文件。')
        return 1
    print('可用的聊天记录：')
    for index, path in enumerate(files, start=1):
        print(f'  [{index}] {path.name}')
    if selection is None:
        try:
            selection = input('请输入要分析的编号: ')
        except EOFError:
            selection = '1'
    try:
        index = int(selection) - 1
        if index < 0 or index >= len(files):
            raise ValueError('无效编号')
        source = files[index]
    except ValueError:
        source = next((path for path in files if selection in path.stem), None)
        if source is None:
            print('未找到匹配的聊天记录。')
            return 1
    artifacts = build_artifacts(source, workspace / 'analysis', include_preview_prompt=include_preview_prompt)
    print(artifacts['stats_summary'])
    for field in ('preproc_file', 'stats_file', 'diagnostics_file'):
        print(f'已生成: {artifacts[field]}')
    if include_preview_prompt:
        print(f'采样分析提示词: {artifacts["prompt_file"]}')
    print('预处理完成。')
    return 0
