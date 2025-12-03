import json
# Removed 'requests' as the file is read locally
import pdfplumber
import re
import io
import os

# --- Configuration ---
# Updated to use the local file path as requested by the user.
LOCAL_PDF_FILENAME = "isinequity_as_of__31__Oct_2025.pdf"
INPUT_FILENAME = "bursa_companies.jsonl"
OUTPUT_FILENAME = "bursa_companies_updated.jsonl"

def read_local_pdf(filepath):
    """Reads a local PDF file into memory (BytesIO)."""
    print(f"Attempting to read local PDF from: {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: Local PDF file not found at '{filepath}'.")
        return None
    try:
        with open(filepath, 'rb') as f:
            pdf_data = f.read()
        print("Successfully read local PDF.")
        return io.BytesIO(pdf_data)
    except Exception as e:
        print(f"Error reading local PDF: {e}")
        return None

def extract_bursa_mapping(pdf_file):
    """
    Parses the Bursa ISIN PDF to build a dictionary mapping, filtering for Ordinary Shares.
    Returns a dict: { 'stock_code': 'STOCK_SHORT_NAME' }
    Fallback dict: { 'company_long_name': 'STOCK_SHORT_NAME' }
    """
    code_map = {}
    name_map = {}
    
    print("Extracting data from PDF (this may take a moment)...")
    
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Extract table data
            table = page.extract_table()
            
            if not table:
                continue
                
            for row in table:
                # Filter out empty rows or headers
                # We expect rows to be [No, Long Name, Short Name, ISIN, Issue Description, ...]
                cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                
                # Ensure we have enough columns for Long Name (idx 1), Short Name (idx 2), ISIN (idx 3), and Issue Description (idx 4)
                if len(cleaned_row) < 5:
                    continue

                long_name = cleaned_row[1]
                short_name = cleaned_row[2]
                isin = cleaned_row[3]
                security_type = cleaned_row[4] # Index 4 is the Issue Description/Security Type

                # Skip header rows
                if "Stock Name" in long_name or "ISIN" in isin or "Issue Description" in security_type:
                    continue
                
                # --- FILTER: ONLY ORDINARY SHARES ---
                if "ORDINARY SHARE" not in security_type.upper():
                    continue
                # ------------------------------------
                
                # Validation: Short Name should not be empty and ISIN should start with MY
                if not short_name or not isin.startswith("MY"):
                    continue

                # 1. Store by Long Name (Exact Match)
                name_map[long_name.upper()] = short_name

                # 2. Derive Stock Code from ISIN for precise matching
                # Pattern: MYLxxxx or MYQxxxx where xxxx is the 4-digit code
                # Example: MYQ0328OO003 -> 0328
                match = re.search(r'MY[LQ](\d{4})', isin)
                if match:
                    derived_code = match.group(1)
                    code_map[derived_code] = short_name

    print(f"Extraction complete. Found {len(code_map)} codes and {len(name_map)} names.")
    return code_map, name_map

def update_jsonl(input_file, output_file, code_map, name_map):
    """Reads JSONL, updates Short Name, and writes to new file."""
    print(f"Processing {input_file}...")
    
    updated_count = 0
    missing_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                stock_code = data.get("stock_code", "")
                long_name = data.get("company_long", "").upper()
                
                current_short = data.get("company_short")

                # Strategy 1: Match by Stock Code (Derived from ISIN) - Most Accurate
                new_short = code_map.get(stock_code)
                
                # Strategy 2: Match by Company Long Name - Fallback
                if not new_short and long_name:
                    new_short = name_map.get(long_name)
                
                # Update if we found a new short name
                if new_short:
                    data["company_short"] = new_short
                    
                    # Also ensure the short name is in the aliases list
                    if "aliases" not in data or data["aliases"] is None:
                        data["aliases"] = []
                    
                    if new_short not in data["aliases"]:
                        data["aliases"].append(new_short)
                        
                    updated_count += 1
                else:
                    missing_count += 1

                # Write the updated line
                outfile.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Updated entries: {updated_count}")
    print(f"Entries processed: {updated_count + missing_count}")
    print(f"Output saved to: {output_file}")

def main():
    # 1. Read Local PDF
    pdf_bytes = read_local_pdf(LOCAL_PDF_FILENAME)
    if not pdf_bytes:
        return

    # 2. Extract
    code_map, name_map = extract_bursa_mapping(pdf_bytes)
    
    # 3. Update
    if os.path.exists(INPUT_FILENAME):
        update_jsonl(INPUT_FILENAME, OUTPUT_FILENAME, code_map, name_map)
    else:
        print(f"Error: Input file '{INPUT_FILENAME}' not found.")

if __name__ == "__main__":
    main()