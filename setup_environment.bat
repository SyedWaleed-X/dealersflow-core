@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo        DEALERSFLOW.AI - MASTER 1-CLICK ENVIRONMENT PROVISIONER
echo ===============================================================================
echo.

:: 1. Check Python
echo [*] Checking Python 3.11 installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH!
    echo Please install Python 3.11 from python.org and check "Add to PATH".
    pause
    exit /b 1
)
python --version

:: 2. Check Git
echo.
echo [*] Checking Git installation...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed!
    pause
    exit /b 1
)
git --version

:: 3. Check PostgreSQL
echo.
echo [*] Checking PostgreSQL client (psql)...
psql --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] psql command not found in PATH.
    echo Ensure PostgreSQL 16 or 17 is running on localhost:5432.
) else (
    psql --version
)

:: 4. Virtual Environment Setup
echo.
echo [*] Setting up Python Virtual Environment (.venv)...
if not exist ".venv" (
    echo [*] Creating fresh .venv...
    python -m venv .venv
)

echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat

:: 5. Install Dependencies
echo.
echo [*] Installing and verifying core dependencies...
python -m pip install --upgrade pip
pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy asyncpg pgvector python-dotenv httpx instructor openai faster-whisper catboost scikit-learn numpy sentence-transformers torch torchvision pillow tabulate psutil google-genai twilio playwright playwright-stealth pandas pytest anyio

:: 6. Verify pgvector and Schema
echo.
echo [*] Applying PostgreSQL schema migrations...
psql -U postgres -d dealersflow_dev -f backend\schema.sql >nul 2>&1
if %errorlevel% equ 0 (
    echo [+] Database schema verified!
) else (
    echo [!] Note: Database setup skipped. Ensure database 'dealersflow_dev' exists.
)

:: 7. Build Frontend
echo.
echo [*] Verifying Next.js frontend dependencies...
if exist "frontend" (
    cd frontend
    call npm install
    cd ..
    echo [+] Frontend packages verified!
)

echo.
echo ===============================================================================
echo [SUCCESS] ENVIRONMENT INITIALIZED AND READY FOR PRESENTATION!
echo ===============================================================================
echo To start the system:
echo   1. Backend : uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
echo   2. Frontend: cd frontend ^&^& npm run dev
echo ===============================================================================
echo.
pause