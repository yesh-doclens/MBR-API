import boto3
import json
import base64
import pandas as pd
import re
import hashlib
from pathlib import Path
from PIL import Image
import fitz
import io
from typing import List, Dict, Any, Tuple, Optional
from rapidfuzz import process, fuzz
from datetime import datetime
from .pricing import enrich_with_medicare_data

# Cache directory paths
# Use /tmp for Vercel serverless (writable), fallback to local for development
import os
if os.environ.get('VERCEL'):
    # Vercel serverless environment - use /tmp (ephemeral storage)
    CACHE_DIR = Path("/tmp/cache")
else:
    # Local development - use project cache directory
    CACHE_DIR = Path(__file__).parent.parent / "cache"

TEXTRACT_CACHE_DIR = CACHE_DIR / "textract"
TXT_CACHE_DIR = CACHE_DIR / "txt"
RECORDS_CACHE_DIR = CACHE_DIR / "records"
ENRICHED_CACHE_DIR = CACHE_DIR / "enriched"
USER_EDITED_CACHE_DIR = CACHE_DIR / "user_edited"

def ensure_cache_dirs():
    """Ensure cache directories exist."""
    TEXTRACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    USER_EDITED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_content_hash(content: bytes) -> str:
    """Generate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()

def get_combined_hash(parts: List[bytes]) -> str:
    """Generate a combined SHA-256 hash of multiple binary parts."""
    combined = b"".join(parts)
    return hashlib.sha256(combined).hexdigest()

def check_cache(content_hash: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """Check if cached results exist for a content hash."""
    textract_path = TEXTRACT_CACHE_DIR / f"{content_hash}.json"
    txt_path = TXT_CACHE_DIR / f"{content_hash}.txt"
    records_path = RECORDS_CACHE_DIR / f"{content_hash}.json"
    enriched_path = ENRICHED_CACHE_DIR / f"{content_hash}.json"
    
    enriched = None
    if enriched_path.exists():
        enriched = json.loads(enriched_path.read_text(encoding='utf-8'))

    records = None
    if records_path.exists():
        records = json.loads(records_path.read_text(encoding='utf-8'))
    
    txt_content = None
    if txt_path.exists():
        txt_content = txt_path.read_text(encoding='utf-8')
        
    textract_json = None
    if textract_path.exists():
        textract_json = json.loads(textract_path.read_text(encoding='utf-8'))
        
    return textract_json, txt_content, records, enriched

def save_to_cache(content_hash: str, textract_response: Dict[str, Any] = None, txt_content: str = None, records: List[Dict[str, Any]] = None, enriched: List[Dict[str, Any]] = None):
    """Save components to cache."""
    ensure_cache_dirs()
    
    if textract_response:
        textract_path = TEXTRACT_CACHE_DIR / f"{content_hash}.json"
        textract_path.write_text(json.dumps(textract_response, indent=2), encoding='utf-8')
    
    if txt_content:
        txt_path = TXT_CACHE_DIR / f"{content_hash}.txt"
        txt_path.write_text(txt_content, encoding='utf-8')
        
    if records:
        records_path = RECORDS_CACHE_DIR / f"{content_hash}.json"
        records_path.write_text(json.dumps(records, indent=2), encoding='utf-8')

    if enriched:
        enriched_path = ENRICHED_CACHE_DIR / f"{content_hash}.json"
        enriched_path.write_text(json.dumps(enriched, indent=2), encoding='utf-8')

def save_user_edits(content_hash: str, records: List[Dict[str, Any]]):
    """Save user-edited records to cache."""
    ensure_cache_dirs()
    user_edited_path = USER_EDITED_CACHE_DIR / f"{content_hash}.json"
    user_edited_path.write_text(json.dumps(records, indent=2), encoding='utf-8')

def check_user_cache(content_hash: str) -> Optional[List[Dict[str, Any]]]:
    """Check if user-edited results exist for a content hash."""
    user_edited_path = USER_EDITED_CACHE_DIR / f"{content_hash}.json"
    if user_edited_path.exists():
        return json.loads(user_edited_path.read_text(encoding='utf-8'))
    return None

def call_textract(image_bytes: bytes) -> Dict[str, Any]:
    """Call AWS Textract analyze_document API."""
    # Use environment variables in Vercel, profile in local development
    if os.environ.get('VERCEL'):
        session = boto3.Session()  # Uses AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY from env
    else:
        session = boto3.Session(profile_name="aidev")
    
    textract_client = session.client(service_name="textract", region_name="us-east-1")
    response = textract_client.analyze_document(
        Document={'Bytes': image_bytes},
        FeatureTypes=['LAYOUT', 'TABLES']
    )
    return response

def get_textract_confidence(response: Dict[str, Any]) -> Dict[str, float]:
    """Calculate average confidence."""
    blocks = response.get('Blocks', [])
    line_confidences = [b.get('Confidence', 0) for b in blocks if b['BlockType'] == 'LINE']
    table_confidences = [b.get('Confidence', 0) for b in blocks if b['BlockType'] == 'TABLE']
    
    avg_line_conf = sum(line_confidences) / len(line_confidences) if line_confidences else 0.0
    avg_table_conf = sum(table_confidences) / len(table_confidences) if table_confidences else avg_line_conf
    
    return {
        'avg_line_confidence': round(avg_line_conf, 2),
        'avg_table_confidence': round(avg_table_conf, 2)
    }

def textract_response_to_txt(response: Dict[str, Any]) -> str:
    """Convert Textract response to structured text."""
    blocks = response.get('Blocks', [])
    block_map = {block['Id']: block for block in blocks}
    output_lines = []
    
    layout_blocks = [b for b in blocks if b['BlockType'].startswith('LAYOUT_')]
    for layout_block in sorted(layout_blocks, key=lambda x: (
        x.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0),
        x.get('Geometry', {}).get('BoundingBox', {}).get('Left', 0)
    )):
        block_type = layout_block['BlockType']
        child_ids = [rel['Ids'] for rel in layout_block.get('Relationships', []) if rel['Type'] == 'CHILD']
        child_ids = [item for sublist in child_ids for item in sublist]
        
        text_parts = []
        for child_id in child_ids:
            child_block = block_map.get(child_id)
            if child_block and child_block['BlockType'] == 'LINE':
                text_parts.append(child_block.get('Text', ''))
        
        if text_parts:
            text = ' '.join(text_parts)
            if block_type == 'LAYOUT_TITLE':
                output_lines.append(f"=== {text} ===")
            elif block_type == 'LAYOUT_HEADER':
                output_lines.append(f"--- {text} ---")
            else:
                output_lines.append(text)
            output_lines.append('')
    
    table_blocks = [b for b in blocks if b['BlockType'] == 'TABLE']
    for table_idx, table_block in enumerate(table_blocks):
        output_lines.append(f"\n=== TABLE {table_idx + 1} ===")
        cell_ids = []
        for rel in table_block.get('Relationships', []):
            if rel['Type'] == 'CHILD':
                cell_ids.extend(rel['Ids'])
        
        cells = [block_map.get(cid) for cid in cell_ids if block_map.get(cid)]
        cells = [c for c in cells if c and c['BlockType'] == 'CELL']
        
        if cells:
            max_row = max(c.get('RowIndex', 1) for c in cells)
            max_col = max(c.get('ColumnIndex', 1) for c in cells)
            table_data = [['' for _ in range(max_col)] for _ in range(max_row)]
            
            for cell in cells:
                row_idx = cell.get('RowIndex', 1) - 1
                col_idx = cell.get('ColumnIndex', 1) - 1
                cell_text_parts = []
                for rel in cell.get('Relationships', []):
                    if rel['Type'] == 'CHILD':
                        for word_id in rel['Ids']:
                            word_block = block_map.get(word_id)
                            if word_block and word_block['BlockType'] == 'WORD':
                                cell_text_parts.append(word_block.get('Text', ''))
                table_data[row_idx][col_idx] = ' '.join(cell_text_parts)
            
            col_widths = [max(len(table_data[r][c]) for r in range(max_row)) for c in range(max_col)]
            col_widths = [max(w, 5) for w in col_widths]
            
            for row_idx, row in enumerate(table_data):
                row_str = ' | '.join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
                output_lines.append(f"| {row_str} |")
                if row_idx == 0:
                    separator = '-+-'.join('-' * w for w in col_widths)
                    output_lines.append(f"+-{separator}-+")
        output_lines.append('')
    
    if not layout_blocks:
        line_blocks = [b for b in blocks if b['BlockType'] == 'LINE']
        for line_block in sorted(line_blocks, key=lambda x: (
            x.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0),
            x.get('Geometry', {}).get('BoundingBox', {}).get('Left', 0)
        )):
            output_lines.append(line_block.get('Text', ''))
    
    return '\n'.join(output_lines)

def parse_txt_with_llm(txt_content: str) -> List[Dict[str, Any]]:
    """Use Claude Sonnet 3.5 to parse TXT content."""
    # Use environment variables in Vercel, profile in local development
    if os.environ.get('VERCEL'):
        session = boto3.Session()  # Uses AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY from env
    else:
        session = boto3.Session(profile_name="aidev")
    
    bedrock_client = session.client(service_name="bedrock-runtime", region_name="us-east-1")

    extraction_prompt = f"""Analyze the following medical bill text extracted via OCR and extract all line items/procedures.

MEDICAL BILL TEXT:
---
{txt_content}
---

For EACH procedure/item in the bill, extract the following information and return as a JSON array:

1. **Date**: The date of service in "YYYY-MM-DD" format. If only month/year is given, use the first day of that month. If no date is found, use "N/A".
2. **Medical Procedure Name**: The full description of the medical procedure or service.
3. **CPT Code**: The CPT (Current Procedural Terminology) code. If it's present in the bill, use that exact code. If not present, determine the most appropriate standard CPT code based on the procedure description. Format as a 5-digit string (e.g., "99213").
4. **CPT Code Source**: Specify whether the CPT code was "extracted" directly from the bill text or "inferred" by you based on the description.
5. **Modifier**: The modifier for the CPT code (e.g., "25", "59", "76"). If present in the bill, use that. If the procedure typically requires a modifier, include it. If no modifier is needed or applicable, use an empty string "".
6. **Quantity**: The quantity/units of the procedure as an integer. Default to 1 if not specified.
7. **Unit Price**: The unit price as a float with 2 decimal places. Remove any currency symbols. If only total is given with quantity > 1, calculate unit price.
8. **Total Price**: The total price as a float with 2 decimal places. This should equal Unit Price × Quantity.
9. **Entity Type**: Classify the item as either "procedure" or "drug". Use "drug" for medications, pharmaceuticals, injections of drugs, infusions, and any medication-related items. Use "procedure" for medical procedures, examinations, consultations, surgeries, lab tests, imaging, and other medical services.
10. **Dosage Value**: For drugs, the numeric value of the dosage mentioned in the name or description (e.g., for "Drug 50mg", the value is 50.0). For procedures, use null or 0.
11. **Dosage Unit**: For drugs, the unit of the dosage (e.g., "MG", "ML", "IU"). Always return in uppercase. For procedures, use an empty string "".
12. **Units from Name**: For drugs, the units from Name is 1 unless some quantity other than the dosage value is mentioned. For example: "Drug 50mg/ 2ml injection" has dosage 50mg and units from name as 2, since default is 1 ml, but this is a 2ml injection. For procedures, use null
13. **Page**: The page number where this item was found in the text (look for the [PAGE X] markers).
14. **Extraction Confidence**: An integer from 0 to 100 representing your confidence in the extraction of THIS specific row.

IMPORTANT FORMATTING RULES:
- Return ONLY a valid JSON array, no additional text
- Dates must be in "YYYY-MM-DD" format or "N/A"
- CPT codes must be 5-character strings
- Modifiers must be strings (empty string if not applicable)
- Quantity must be an integer
- Unit Price and Total Price must be floats with 2 decimal places
- Entity Type must be exactly "procedure" or "drug"
- Dosage Value must be a float or null
- Dosage Unit must be an uppercase string or ""
- If a value cannot be determined, use reasonable defaults or "N/A" for strings, 0 for numbers

Now analyze the medical bill text and extract ALL line items:"""

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": extraction_prompt}],
    }

    response = bedrock_client.invoke_model(
        modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    response_text = response_body["content"][0]["text"]

    json_match = re.search(r"\[[\s\S]*\]", response_text)
    if json_match:
        return json.loads(json_match.group())
    raise ValueError("Could not parse JSON from model response")

def parse_date(date_value) -> str:
    if date_value is None or str(date_value).strip().upper() in ["N/A", "NA", ""]:
        return "N/A"
    date_str = str(date_value).strip()
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y"]
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return "N/A"

def format_cpt_code(cpt_value) -> str:
    if not cpt_value: return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(cpt_value).strip())

def parse_quantity(qty_value) -> int:
    try:
        return max(1, int(float(str(qty_value).replace(",", ""))))
    except:
        return 1

def parse_price(price_value) -> float:
    try:
        return round(float(str(price_value).replace("$", "").replace(",", "").strip()), 2)
    except:
        return 0.0

def parse_and_format_data(raw_data: List[Dict[str, Any]], document_name: str) -> pd.DataFrame:
    formatted_records = []
    for record in raw_data:
        entity_type = str(record.get("Entity Type", "procedure")).lower().strip()
        formatted_record = {
            "Document Name": document_name,
            "Date": parse_date(record.get("Date", "N/A")),
            "Medical Procedure Name": str(record.get("Medical Procedure Name", "")).strip(),
            "CPT Code": format_cpt_code(record.get("CPT Code", "")),
            "CPT Code Source": str(record.get("CPT Code Source", "extracted")).lower().strip(),
            "Modifier": str(record.get("Modifier", "")).strip(),
            "Quantity": parse_quantity(record.get("Quantity", 1)),
            "Unit Price": parse_price(record.get("Unit Price", 0)),
            "Total Price": parse_price(record.get("Total Price", 0)),
            "Entity Type": entity_type if entity_type in ["procedure", "drug"] else "procedure",
            "Dosage Value": record.get("Dosage Value"),
            "Dosage Unit": str(record.get("Dosage Unit", "")).strip().upper(),
            "Units from Name": record.get("Units from Name"),
            "Page": pd.to_numeric(record.get("Page", 1), errors='coerce'),
            "Extraction Confidence": pd.to_numeric(record.get("Extraction Confidence", 0), errors='coerce'),
            "Highlight Geometry": record.get("Highlight Geometry")
        }
        formatted_records.append(formatted_record)

    df = pd.DataFrame(formatted_records)
    df = enrich_with_medicare_data(df)
    return df

def extract_medical_bill_data(image_bytes: bytes, file_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract data from a medical bill, using content-based caching for speed."""
    ensure_cache_dirs()
    content_hash = get_content_hash(image_bytes)
    
    # 1. Check for complete enriched records (Final state)
    cached_textract, cached_txt, cached_records, cached_enriched = check_cache(content_hash)
    
    if cached_enriched:
        conf = get_textract_confidence(cached_textract) if cached_textract else {}
        return cached_enriched, {'cached': True, 'is_enriched': True, 'content_hash': content_hash, **conf}

    # 2. Check for raw LLM records
    if cached_records:
        conf = get_textract_confidence(cached_textract) if cached_textract else {}
        return cached_records, {'cached': True, 'is_enriched': False, 'content_hash': content_hash, **conf}
    
    # 3. Check for cached OCR text
    if cached_txt:
        records = parse_txt_with_llm(cached_txt)
        save_to_cache(content_hash, records=records)
        conf = get_textract_confidence(cached_textract) if cached_textract else {}
        return records, {'cached': True, 'ocr_cached': True, 'is_enriched': False, 'content_hash': content_hash, **conf}
    
    # 4. Full processing
    response = call_textract(image_bytes)
    _, geometries = get_textract_blocks_for_pages(response, [1]) # Assume page 1 for single image/PDF
    txt_content = textract_response_to_txt(response)
    records = parse_txt_with_llm(txt_content)
    
    # Match records to geometries for highlighting
    records = match_records_to_geometries(records, geometries)
    
    # Save components to cache (except enriched which happens after pricing)
    save_to_cache(content_hash, textract_response=response, txt_content=txt_content, records=records)
    
    return records, {'cached': False, 'is_enriched': False, 'content_hash': content_hash, **get_textract_confidence(response)}

def filter_medical_bill_pages(csv_df: pd.DataFrame) -> List[int]:
    """Filter CSV to get page numbers classified as medical bills."""
    if 'Page Classification' not in csv_df.columns or 'Number' not in csv_df.columns:
        return []
    
    # Target classification used in original app
    target_class = "Bills – Medical, Pharmacy"
    medical_bill_pages = csv_df[csv_df['Page Classification'] == target_class]['Number'].tolist()
    return [int(p) for p in medical_bill_pages]

def get_textract_blocks_for_pages(textract_json: Dict[str, Any], pages: List[int]) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract LINE blocks and their geometry for specific pages from Textract JSON."""
    all_text = []
    all_geometries = []
    
    blocks = textract_json.get('Blocks', [])
    
    # Process page by page to insert markers
    for page_num in sorted(pages):
        all_text.append(f"\n[PAGE {page_num}]\n")
        for block in blocks:
            # Handle cases where 'Page' might be missing (default to 1)
            b_page = block.get('Page', 1)
            if block.get('BlockType') == 'LINE' and b_page == page_num:
                text = block.get('Text', '')
                geometry = block.get('Geometry', {}).get('BoundingBox', {})
                
                all_text.append(text)
                all_geometries.append({
                    'Text': text,
                    'Page': page_num,
                    'Geometry': json.dumps(geometry) # Store as JSON string for easy storage/table use
                })
            
    return "\n".join(all_text), all_geometries

def match_records_to_geometries(records: List[Dict[str, Any]], geometries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Associate LLM records with Textract geometries using fuzzy matching."""
    for record in records:
        try:
            page = int(float(str(record.get("Page", 1))))
        except:
            page = 1
        
        proc_name = str(record.get("Medical Procedure Name", ""))
        if not proc_name: continue
        
        # Filter geometries for the specific page
        page_geoms = [g for g in geometries if g["Page"] == page]
        if not page_geoms: continue
        
        # Find best match for the procedure name among Textract LINE blocks
        choices = [g["Text"] for g in page_geoms]
        best_match = process.extractOne(proc_name, choices, scorer=fuzz.partial_ratio)
        
        if best_match and best_match[1] > 70: # 70 threshold for partial matches
            match_text = best_match[0]
            # Find the geometry that produced this text
            for pg in page_geoms:
                if pg["Text"] == match_text:
                    record["Highlight Geometry"] = pg["Geometry"]
                    break
            
    return records

def process_batch_pdf_data(pdf_bytes: bytes, csv_bytes: bytes, json_bytes: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process a batch PDF using classification CSV and Textract JSON blocks."""
    content_hash = get_combined_hash([pdf_bytes, csv_bytes, json_bytes])
    
    # 1. Check if cached results exist
    cached_textract, cached_txt, cached_records, cached_enriched = check_cache(content_hash)
    
    if cached_enriched:
        return cached_enriched, {'cached': True, 'is_enriched': True, 'content_hash': content_hash}
        
    if cached_records:
        return cached_records, {'cached': True, 'is_enriched': False, 'content_hash': content_hash}

    # 2. Load CSV and JSON
    csv_df = pd.read_csv(io.BytesIO(csv_bytes))
    textract_json = json.loads(json_bytes.decode('utf-8'))
    
    # 3. Filter pages
    bill_pages = filter_medical_bill_pages(csv_df)
    if not bill_pages:
        return [], {"error": "No pages classified as 'Bills – Medical, Pharmacy' found.", "content_hash": content_hash}
        
    # 4. Extract text for those pages
    raw_text, geometries = get_textract_blocks_for_pages(textract_json, bill_pages)
    
    # 5. Parse with LLM
    records = parse_txt_with_llm(raw_text)
    
    # 6. Match records to geometries for highlighting
    records = match_records_to_geometries(records, geometries)
    
    # 7. Save raw LLM records to cache
    save_to_cache(content_hash, records=records, txt_content=raw_text)
    
    metadata = {
        'processed_pages': bill_pages,
        'page_count': len(bill_pages),
        'total_pages_in_pdf': int(csv_df['Number'].max()) if 'Number' in csv_df.columns else 0,
        'content_hash': content_hash,
        'cached': False
    }
    
    return records, metadata

def render_pdf_page_with_highlight(pdf_bytes: bytes, page_number: int, geometry: Dict[str, Any] = None) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_number - 1)
    if geometry:
        rect = page.rect
        w, h = rect.width, rect.height
        l, t = geometry.get('Left', 0) * w, geometry.get('Top', 0) * h
        width, height = geometry.get('Width', 0) * w, geometry.get('Height', 0) * h
        annot = page.add_rect_annot(fitz.Rect(l, t, l + width, t + height))
        annot.set_colors(fill=(0.1, 0.5, 1.0))
        annot.set_opacity(0.3)
        annot.update()
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    return pix.tobytes()

def get_pdf_with_highlight_base64(pdf_bytes: bytes, page_number: int, geometry: Dict[str, Any] = None) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if geometry:
        page = doc.load_page(page_number - 1)
        rect = page.rect
        w, h = rect.width, rect.height
        l, t = geometry.get('Left', 0) * w, geometry.get('Top', 0) * h
        width, height = geometry.get('Width', 0) * w, geometry.get('Height', 0) * h
        annot = page.add_highlight_annot(fitz.Rect(l, t, l + width, t + height))
        annot.set_colors(stroke=(0.1, 0.5, 1.0))
        annot.update()
    pdf_out = doc.write(clean=True, deflate=True)
    return base64.b64encode(pdf_out).decode('utf-8')
