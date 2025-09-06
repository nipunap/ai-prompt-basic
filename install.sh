#!/bin/bash

# AI Conversation Memory System - Installation Script
# This script automates the setup process for the AI prompt application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Welcome message
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              AI Conversation Memory System                   ║"
echo "║                   Installation Script                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

print_status "Starting installation process..."

# Check Python installation
print_status "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is required but not installed"
    print_status "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Check Python version (require 3.8+)
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8+ is required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ] || [ ! -d "backend" ]; then
    print_error "Please run this script from the ai-prompt-basic directory"
    print_status "Make sure you have cloned the repository and are in the project root"
    exit 1
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
if command_exists pip3; then
    pip3 install -r requirements.txt
    print_success "Dependencies installed successfully"
else
    print_error "pip3 not found. Please install pip"
    exit 1
fi

# Create models directory
print_status "Creating models directory..."
mkdir -p models
print_success "Models directory created"

# Create .env file if it doesn't exist
if [ ! -f "backend/.env" ]; then
    print_status "Creating environment configuration file..."
    cat > backend/.env << EOF
# AI Model Configuration
MODEL_PATH=./models/llama-2-7b-chat.q4.gguf

# Model Parameters
MAX_TOKENS=2000
TEMPERATURE=0.7
TOP_P=0.95
EOF
    print_success "Environment file created at backend/.env"
    print_warning "Please update MODEL_PATH in backend/.env with your actual model file path"
else
    print_status "Environment file already exists"
fi

# Check for model file
print_status "Checking for LLaMA model..."
if [ -f "models/llama-2-7b-chat.q4.gguf" ] || [ -f "models/llama-2-7b-chat.q5.gguf" ]; then
    print_success "Model file found"
else
    print_warning "No model file found in models/ directory"
    echo
    print_status "To download a model, you can use:"
    echo -e "${YELLOW}# For 4GB model (faster, less accurate):${NC}"
    echo "wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O models/llama-2-7b-chat.q4.gguf"
    echo
    echo -e "${YELLOW}# For 5GB model (slower, more accurate):${NC}"
    echo "wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf -O models/llama-2-7b-chat.q5.gguf"
    echo
fi

# Test installation
print_status "Testing installation..."
cd backend

# Test imports
python3 -c "
try:
    import fastapi
    import uvicorn
    import pydantic
    import sqlite3
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
" || {
    print_error "Package import test failed"
    exit 1
}

print_success "Installation test passed"

# Installation complete
echo
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 Installation Complete! 🎉                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo
print_status "Next steps:"
echo "1. Download a LLaMA model (see instructions above)"
echo "2. Update MODEL_PATH in backend/.env"
echo "3. Start the server:"
echo -e "   ${YELLOW}cd backend${NC}"
echo -e "   ${YELLOW}uvicorn app:app --reload${NC}"
echo "4. Open http://localhost:8000 in your browser"
echo

print_status "For more information, see README.md"
print_success "Happy chatting with your AI! 🤖"
