@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ==============================================================
rem  源码方式启动（开发/调试用）
rem  普通用户请直接双击 dist\AI求职助手.exe，不需要这个脚本
rem ==============================================================

set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [错误] 没找到 Python。
    echo        请先安装依赖：pip install -r requirements.txt
    pause
    exit /b 1
)

echo ==============================================================
echo   AI 求职助手（源码方式启动）
echo   Python : %PY%
echo   启动后会自动打开浏览器，按 Ctrl+C 退出
echo ==============================================================
echo.

"%PY%" main.py %*

echo.
echo [已退出]
pause
