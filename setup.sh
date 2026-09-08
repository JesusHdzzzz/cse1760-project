#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

echo "Setting up CSE1760 project environment..."

# Make sure Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# Make sure Python venv support is available
if ! python3 -m venv --help &> /dev/null; then
    echo "Error: python3 venv support is not installed."
    echo "Install it with your system package manager, for example:"
    echo "  sudo apt install python3-venv"
    exit 1
fi

# Make sure Java is available for H2O
if ! command -v java &> /dev/null; then
    echo "Error: Java is not installed or not on PATH."
    echo "H2O requires Java. Install OpenJDK with your system or environment manager."
    echo "For Conda environments, for example: conda install -c conda-forge openjdk"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists."
fi

# Activate environment
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies..."
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo ""
echo "Setup complete."
echo "Activate the environment with:"
echo "source .venv/bin/activate"
