@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ==============================================================
rem  源码方式启动（开发/调试用）
rem
rem  普通用户不需要这个脚本 —— 请直接去 Releases 页面下载
rem  AI求职助手.exe，双击即用，不需要装 Python：
rem      https://github.com/want-get/ai-resume-helper/releases/latest
rem ==============================================================

set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [错误] 没找到 Python。
    echo.
    echo   普通用户：请直接下载 exe，双击即用，不需要 Python：
    echo     https://github.com/want-get/ai-resume-helper/releases/latest
    echo.
    echo   开发者：请先安装 Python 3.11+，然后执行
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

rem 先确认依赖装好了，否则报错信息对新手很难懂
"%PY%" -c "import fastapi, uvicorn, chromadb, sqlalchemy" >nul 2>nul
if errorlevel 1 (
    echo [错误] Python 找到了，但依赖没装全。
    echo.
    echo   请在当前目录执行：
    echo     "%PY%" -m pip install -r requirements.txt
    echo.
    echo   或者直接下载 exe，双击即用，不需要 Python：
    echo     https://github.com/want-get/ai-resume-helper/releases/latest
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
