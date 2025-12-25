# Water Meter OCR

AI-powered water meter reading extraction from handwritten tables using OpenAI API. Converts PDF documents to CSV with automatic validation

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview
This project automates the extraction of water meter readings from scanned PDF documents containing handwritten data. It uses OpenAI's Vision API (GPT-4o) combined with OpenCV for image processing to accurately read both printed and handwritten numbers from tabular data.

**Key Features:**
- AI-powered OCR using OpenAI API
- Handles handwritten numbers with high accuracy
- Automatic PDF to CSV conversion
- Built-in validation and error detection
- Image enhancement (contrast, sharpness, rotation)
- Warns about suspicious readings for manual verification

## Use Case
Designed for water utility companies and meter readers who need to digitize handwritten meter readings from paper forms. The system reads tables with columns like:
- **Šifra** (ID code)
- **Novi status** (New status - decimal)
- **Staro stanje** (Old reading - printed)
- **Novo stanje** (New reading - handwritten)

## Quick Start

### Prerequisites
- Python 3.12+
- OpenAI API key with GPT-4o access
- System dependencies: Poppler, Tesseract

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/Haris0059water-meter-ocr.git
cd water-meter-ocr
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install system dependencies**

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
```

**macOS:**
```bash
brew install poppler tesseract
```

4. **Configure environment variables**

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
PDF_PATH=test.pdf
OUTPUT_CSV=results.csv
TEMP_DIR=output_pages
DPI=400
```

5. **Run the extraction**
```bash
python main.py
```

## Configuration
### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key (required) |
| `PDF_PATH` | `test.pdf` | Input PDF file path |
| `OUTPUT_CSV` | `results.csv` | Output CSV file name |
| `TEMP_DIR` | `output_pages` | Directory for temporary images |
| `DPI` | `400` | Scan resolution (300-600 recommended) |

### Scanning Tips
For best results:
- Use **black & white** scanning mode
- Ensure good lighting and flat surface
- Scan at **300-400 DPI**
- Keep handwriting clear and legible
- Avoid shadows and wrinkles

## Output Format
The script generates a CSV file with the following columns:

```csv
sifra,novi_status,staro_stanje,novo_stanje,status
00020011,0.0,3306,3326,Ispravan
00020012,6.0,5236,5256,Ispravan
00020013,0.0,2538,2572,Ispravan
```

**Columns:**
- `sifra`: Meter ID code
- `novi_status`: New status value (decimal)
- `staro_stanje`: Old meter reading
- `novo_stanje`: New meter reading (handwritten)
- `status`: Validation status (`Ispravan` = valid, `Neispravan` = invalid)

## How It Works
1. **PDF Conversion**: Converts PDF pages to high-resolution images
2. **Image Enhancement**: Rotates, sharpens, and enhances contrast
3. **AI Extraction**: Sends enhanced image to OpenAI Vision API
4. **Validation**: Checks for suspicious readings (large differences, digit mismatches)
5. **CSV Export**: Saves validated data to CSV file

### Validation Rules
The system automatically validates readings and flags issues:
- **Large difference warning**: If `novo_stanje` differs from `staro_stanje` by >500 units
- **Digit count mismatch**: If handwritten number has more digits than expected
- **Invalid status**: If new reading < old reading (meter rollback)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Contributions are welcome! Please feel free to submit a Pull Request.

**Note:** This project requires an OpenAI API key and will incur costs based on usage. Always monitor your API usage and set spending limits.
