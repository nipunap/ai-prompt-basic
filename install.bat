@echo off
REM AI Conversation Memory System - Windows Installation Script

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              AI Conversation Memory System                   ║
echo ║                Windows Installation Script                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo [INFO] Starting installation process...

REM Check Python installation
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is required but not installed
    echo [INFO] Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [SUCCESS] Python found

REM Check if we're in the correct directory
if not exist "requirements.txt" (
    echo [ERROR] Please run this script from the ai-prompt-basic directory
    pause
    exit /b 1
)

REM Install Python dependencies
echo [INFO] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [SUCCESS] Dependencies installed successfully

REM Create models directory
echo [INFO] Creating models directory...
if not exist "models" mkdir models
echo [SUCCESS] Models directory created

REM Create .env file if it doesn't exist
if not exist "backend\.env" (
    echo [INFO] Creating environment configuration file...
    (
        echo # AI Model Configuration
        echo MODEL_PATH=./models/llama-2-7b-chat.q4.gguf
        echo.
        echo # Model Parameters
        echo MAX_TOKENS=2000
        echo TEMPERATURE=0.7
        echo TOP_P=0.95
    ) > backend\.env
    echo [SUCCESS] Environment file created
    echo [WARNING] Please update MODEL_PATH in backend\.env
) else (
    echo [INFO] Environment file already exists
)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                 Installation Complete! 🎉                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo [INFO] Next steps:
echo 1. Download a LLaMA model
echo 2. Update MODEL_PATH in backend\.env
echo 3. Start the server: cd backend ^&^& uvicorn app:app --reload
echo 4. Open http://localhost:8000 in your browser
echo.
pause
