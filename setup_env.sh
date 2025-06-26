#!/bin/bash

# Environment Setup Script
# This script installs uv and sets up a Python virtual environment

set -e  # Exit on any error

echo "Starting environment setup..."

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source the shell to make uv available in current session
# This handles the case where uv is installed to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

# Create virtual environment
echo "Creating virtual environment..."
uv venv
source .venv/bin/activate

# Sync dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    uv pip sync requirements.txt
    echo "Dependencies installed successfully!"
else
    echo "Warning: requirements.txt not found in current directory"
    echo "Please ensure requirements.txt is present before running this script"
    exit 1
fi

echo "Environment setup completed successfully!"
echo "To activate the virtual environment, run: source .venv/bin/activate"