@echo off
echo ============================================================
echo   Aviation RAG - Setup Script
echo ============================================================
echo.

:: Create virtual environment
echo [1/4] Creating virtual environment...
py -m venv venv
call venv\Scripts\activate

:: Install dependencies
echo.
echo [2/4] Installing dependencies (this may take a few minutes)...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

:: Setup .env
echo.
echo [3/4] Setting up environment...
if not exist .env (
    copy .env.example .env
    echo Created .env file - please edit it and add your GROQ_API_KEY
) else (
    echo .env file already exists
)

:: Create data directories
echo.
echo [4/4] Creating directory structure...
if not exist data\raw\dgca_cars mkdir data\raw\dgca_cars
if not exist data\raw\aai_circulars mkdir data\raw\aai_circulars
if not exist data\raw\icao mkdir data\raw\icao
if not exist data\raw\notams mkdir data\raw\notams
if not exist data\raw\aip mkdir data\raw\aip
if not exist data\processed mkdir data\processed
if not exist vectorstore mkdir vectorstore

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Edit .env and add your GROQ_API_KEY
echo   2. Place PDFs in data\raw\ subdirectories
echo   3. Run: py ingest.py
echo   4. Run: py -m streamlit run app.py
echo.
pause
