@echo off
chcp 65001 >nul
title 鐓х墖搴熺墖妫€娴嬪伐鍏?v4.4
echo.
echo   姝ｅ湪鍚姩鐓х墖搴熺墖妫€娴嬪伐鍏?v4.4...
echo.

py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [閿欒] 鏈壘鍒?Python
    echo   涓嬭浇鍦板潃锛歨ttps://www.python.org/downloads/
    pause
    exit /b 1
)

py "%~dp0photo_checker.py"
if %errorlevel% neq 0 (
    echo.
    echo   [閿欒] 绋嬪簭寮傚父閫€鍑猴紝涓婃柟淇℃伅鍙埅鍥惧弽棣?    pause
)

