@echo off
chcp 65001 >nul
title 照片废片检测工具 v3.0
echo.
echo   正在启动照片废片检测工具 v3.0...
echo.

py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [错误] 未找到 Python
    echo   下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

py "%~dp0photo_checker.py"
if %errorlevel% neq 0 (
    echo.
    echo   [错误] 程序异常退出，上方信息可截图反馈
    pause
)