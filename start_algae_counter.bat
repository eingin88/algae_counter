@echo off
REM ===============================
REM 啟動專案環境
REM ===============================

REM 切到此 bat 檔所在的資料夾
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
echo You can now run:
echo   run.py
echo   python -m cellpose # can open cellpose GUI
echo.


REM 保持視窗不關閉
cmd
