@echo off
REM ===============================
REM 啟動專案環境
REM ===============================

REM 切到此bat檔所在的資料夾
cd /d %~dp0

echo.
echo [INFO] Current directory:
cd

echo.
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo.
echo [OK] Virtual environment activated.
echo.
echo.
run.py

REM 保持視窗不關閉
cmd
