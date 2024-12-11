#!/bin/bash

# Directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Reinstall tkinter
echo "Reinstalling tkinter..."
brew reinstall python-tk@3.13

# Install required packages
echo "Installing required packages..."
pip3 install --break-system-packages pillow pytesseract pandas opencv-python matplotlib

# Run the finance scanner
python3 finance_scanner.py
