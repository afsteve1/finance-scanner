# Finance Statement Scanner

This application helps you convert financial statements into CSV format using screenshots and OCR technology.

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine installed on your system

### Installing Tesseract

On macOS:
```bash
brew install tesseract
```

## Installation

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python finance_scanner.py
```

## Usage

1. Click "Take Screenshot" to capture your financial statement
2. Or use "Load Image" to load an existing image
3. Click "Process Image" to convert the statement to CSV

The application will create a CSV file with the extracted data in the same directory.

## Features

- Screenshot capture
- Image loading
- OCR text extraction
- CSV conversion
- Simple GUI interface
