@echo off
cd /d "%~dp0"

echo ==========================================
echo Customer Support Ticket System
echo ==========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting application...
echo.

python -m streamlit run src\predict.py

pause