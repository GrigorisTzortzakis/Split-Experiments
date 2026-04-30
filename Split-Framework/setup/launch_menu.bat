@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "VENV_PYTHON=%~dp0..\..\..\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
	"%VENV_PYTHON%" experiment_menu.py
) else (
	python experiment_menu.py
	if errorlevel 9009 py experiment_menu.py
)
if errorlevel 1 (
	echo.
	echo Launcher failed. Read the error above.
	pause
)
endlocal