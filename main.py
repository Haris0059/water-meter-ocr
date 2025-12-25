from pdf2image import convert_from_path
import cv2
import numpy as np
import csv
import base64
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageEnhance


load_dotenv()

# Increase Pillow's image size limit to handle high-DPI PDFs
Image.MAX_IMAGE_PIXELS = None

# Configuration
PDF_PATH = os.getenv("PDF_PATH", "test.pdf")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "results.csv")
TEMP_DIR = os.getenv("TEMP_DIR", "output_pages")
DPI = int(os.getenv("DPI", "300"))

# Initialize OpenAI client
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")

client = OpenAI(api_key=API_KEY)
os.makedirs(TEMP_DIR, exist_ok=True)

# 1. Convert PDF to High DPI Images
print("Converting PDF to images...")
pages = convert_from_path(PDF_PATH, dpi=DPI)
image_paths = []
for i, p in enumerate(pages):
    path = f"{TEMP_DIR}/page_{i}.png"
    p.save(path, "PNG")
    image_paths.append(path)
    print(f"  Saved page {i+1}/{len(pages)}")

# 2. Image Enhancement
def enhance_image(img_path):
    """Enhance image quality for better OCR"""
    pil_img = Image.open(img_path)
    
    pil_img = pil_img.rotate(90, expand=True) # Rotate 90 degrees counter-clockwise
    print("  Rotated 90° counter-clockwise")

    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.5)  # Increase contrast
    

    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(2.0)  # Increase sharpness

    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    enhanced_path = f"{TEMP_DIR}/enhanced_table.png" 
    pil_img.save(enhanced_path) # Save Enhanced Image
    print(f"  Saved enhanced image to {enhanced_path}")
    
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) # Convert to OpenCV format
    
    return img_cv, enhanced_path

# 3. AI OCR - Process entire table
def encode_img_path(img_path):
    """Encode image file to base64"""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def extract_all_rows(img_path):
    """Extract all rows from entire table using OpenAI Vision API"""
    img64 = encode_img_path(img_path)
    
    prompt = """
    This is a water meter reading table in Serbian/Bosnian language (scanned in black and white).

    The table has these columns (from left to right):
    1. Redni broj (row number) - ignore this
    2. Šifra (code/ID) - printed numbers (8 digits like 00020011)
    3. Tip (type) - ignore this
    4. Korisnik/Adresa (user/address) - ignore this
    5. Novi status (new status) - decimal number, PRINTED
    6. Staro stanje (old reading) - integer, PRINTED (4 digits typically)
    7. Novo stanje (new reading) - integer, HANDWRITTEN (rightmost column, 4 digits typically)
    
    CRITICAL INSTRUCTIONS FOR READING HANDWRITTEN NUMBERS:
    
    1. COMMON MISTAKES TO AVOID:
       - Don't confuse 1 and 7
       - Don't confuse 0 and 6
       - Don't confuse 3 and 8
       - Don't confuse 5 and S
       - Don't confuse 2 and Z
    
    2. IGNORE TABLE LINES:
       - Table has horizontal lines between rows
       - These lines may touch or cross numbers
       - DO NOT interpret lines as part of digits
       - If a number looks like "3105" but the middle digit has a line through it, it's probably "305" not "3105"
       - Focus on the actual handwritten strokes, not the printed table lines
    
    3. VALIDATION LOGIC:
       - novo_stanje should be CLOSE TO staro_stanje (usually within 0-200 units difference)
       - If your reading shows novo_stanje is 1000+ units away from staro_stanje, you probably misread it
       - Water meters don't jump by thousands of units
       - Most readings differ by 0-100 units
    
    4. NUMBER LENGTH:
       - staro_stanje is typically 4 digits (e.g., 3306, 5236, 2538)
       - novo_stanje should also be 4 digits typically
       - If you read a 5-digit number (like 13106), check if it should be 4 digits (like 3106 or 1306)
    
    Extract ALL rows of data you can read. For each row, extract:
    - sifra: the code/ID (8 digits, like 00020011, 00020012)
    - novi_status: decimal number from the printed column
    - staro_stanje: old reading integer from the printed column (4 digits typically)
    - novo_stanje: NEW reading integer from the HANDWRITTEN column (should be close to staro_stanje)
    
    Return a JSON array with ALL rows:
    [
      {"sifra": "00020011", "novi_status": "0.0", "staro_stanje": "3306", "novo_stanje": "3326"},
      {"sifra": "00020012", "novi_status": "6.0", "staro_stanje": "5236", "novo_stanje": "5256"}
    ]
    
    FINAL CHECK BEFORE RETURNING:
    - Does novo_stanje make sense compared to staro_stanje?
    - Are both numbers similar length (usually 4 digits)?
    - Did you accidentally include a table line as a digit?
    
    Return ONLY the JSON array, no explanation.
    """
    
    print("Sending entire table to AI for extraction...")
    print("(This may take 10-20 seconds...)")
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img64}"
                        }
                    }
                ]
            }]
        )
        
        content = response.choices[0].message.content.strip()
        
        print(f"  AI returned {len(content)} characters")

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        print(f"  First 100 chars: {content[:100]}")
        
        data_array = json.loads(content)
        
        if not isinstance(data_array, list):
            print("  ✗ AI did not return an array")
            return []
        
        results = []
        warnings = []
        
        for idx, row_data in enumerate(data_array):
            try:
                # Validate and convert types
                row = {
                    "sifra": str(row_data.get("sifra", "0")),
                    "novi_status": float(str(row_data.get("novi_status", 0.0)).replace(",", ".")),
                    "staro_stanje": int(float(str(row_data.get("staro_stanje", 0)).replace(",", "."))),
                    "novo_stanje": int(float(str(row_data.get("novo_stanje", 0)).replace(",", ".")))
                }
                
                difference = abs(row["novo_stanje"] - row["staro_stanje"])
                
                if difference > 500:
                    warning_msg = f"⚠️  Row {idx+1} (šifra {row['sifra']}): Large difference detected!"
                    warning_msg += f"\n    Staro: {row['staro_stanje']}, Novo: {row['novo_stanje']} (diff: {difference})"
                    warning_msg += f"\n    This might be a misread. Please verify manually."
                    warnings.append(warning_msg)
                
                staro_len = len(str(row["staro_stanje"]))
                novo_len = len(str(row["novo_stanje"]))
                if novo_len > staro_len + 1:
                    warning_msg = f"⚠️  Row {idx+1} (šifra {row['sifra']}): Digit count mismatch!"
                    warning_msg += f"\n    Staro: {row['staro_stanje']} ({staro_len} digits), Novo: {row['novo_stanje']} ({novo_len} digits)"
                    warning_msg += f"\n    Novo stanje might have extra digit from table line. Please verify."
                    warnings.append(warning_msg)
                
                if row["novo_stanje"] < row["staro_stanje"]:
                    row["novo_stanje"] = row["staro_stanje"]
                    row["status"] = "Neispravan"
                else:
                    row["status"] = "Ispravan"
                
                results.append(row)
                
            except Exception as e:
                print(f"  ✗ Error processing row: {e}")
                continue
        
        if warnings:
            print("\n" + "="*60)
            print("VALIDATION WARNINGS - Please verify these rows manually:")
            print("="*60)
            for warning in warnings:
                print(warning)
            print("="*60 + "\n")
        
        return results
        
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse error: {e}")
        print(f"  Raw response: {content[:200]}...")
        return []
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []

# 4. Process & save CSV
print("\nProcessing tables...")
all_results = []

for img_idx, img_path in enumerate(image_paths):
    print(f"\nProcessing page {img_idx + 1}/{len(image_paths)}...")
    
    enhanced_img, enhanced_path = enhance_image(img_path)
    
    print(f"Check enhanced image: xdg-open {enhanced_path}")
    print("Press Enter to continue with AI extraction (or Ctrl+C to cancel)...")
    input()  # Wait for user confirmation
    
    results = extract_all_rows(enhanced_path)
    
    if results:
        print(f"  ✓ Extracted {len(results)} rows from this page")
        all_results.extend(results)
    else:
        print(f"  ✗ No rows extracted from this page")

if all_results:
    print(f"\nWriting {len(all_results)} rows to CSV...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sifra", "novi_status", "staro_stanje", "novo_stanje", "status"])
        for r in all_results:
            writer.writerow([
                r["sifra"],
                r["novi_status"],
                r["staro_stanje"],
                r["novo_stanje"],
                r["status"]
            ])
    
    print(f"\n✓ DONE → CSV generated: {OUTPUT_CSV}")
    print(f"Total rows extracted: {len(all_results)}")
    
    print("\nFirst 3 rows:")
    for i, row in enumerate(all_results[:3]):
        print(f"  {i+1}. Šifra: {row['sifra']}, Novo stanje: {row['novo_stanje']}, Status: {row['status']}")
else:
    print("\n✗ No data extracted. Check the enhanced image quality.")
    print("You may need to:")
    print("  - Rescan the PDF at higher quality")
    print("  - Adjust DPI in .env (try DPI=400)")
    print("  - Use gpt-4o for better compatibility")