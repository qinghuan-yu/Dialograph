@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"
title 聊天记录分析一键启动

echo ================================================
echo 聊天记录分析一键启动
echo ================================================

set "PYTHON_CMD=.venv\Scripts\python.exe"
if exist "%PYTHON_CMD%" (
    "%PYTHON_CMD%" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo [提示] .venv\Scripts\python.exe 存在但无法启动，尝试系统 Python...
        set "PYTHON_CMD="
    )
) else (
    set "PYTHON_CMD="
)

if not defined PYTHON_CMD (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
    )
)

if not defined PYTHON_CMD (
    echo [错误] 未找到可用的 Python。
    echo 请重新创建 .venv，或安装可从命令行启动的 Python 3。
    pause
    exit /b 1
)

if not exist ".env" (
    echo [错误] 未找到工作区根目录 .env
    echo 请先参考 .env.example 填写 LLM_API_BASE、LLM_API_KEY、LLM_MODEL。
    pause
    exit /b 1
)

echo [1/2] 清空旧的 analysis 产物...
if exist "analysis" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem 'analysis' -Force | Remove-Item -Recurse -Force"
)

echo [2/2] 开始处理 talks 下全部 json 文件...
call %PYTHON_CMD% "agent\run_workflow.py" --force
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
    echo 分析完成。
    echo 输出目录：analysis
    echo 关系建模文件：analysis\分析_对象名.md
    echo 人物侧写文件：analysis\人物侧写_对象名.md
) else (
    echo 分析未完全成功，退出码：%EXIT_CODE%
    echo 请查看上方输出定位失败的对象或接口问题。
)

pause
exit /b %EXIT_CODE%
