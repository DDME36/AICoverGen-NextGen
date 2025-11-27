@echo off
echo ============================================================
echo    AICoverGen-NextGen
echo    AI Cover Song Generator
echo ============================================================
echo.

cd /d "%~dp0"

REM Activate virtual environment if exists
if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
)

REM Run WebUI
python src/webui.py %*

pause
