from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import pandas as pd
import io
import base64
from .processor import (
    extract_medical_bill_data,
    parse_and_format_data,
    render_pdf_page_with_highlight,
    get_pdf_with_highlight_base64,
    save_to_cache,
    process_batch_pdf_data,
    save_user_edits,
    check_user_cache,
    get_content_hash
)
from pathlib import Path
import os

app = FastAPI(title="Medical Bill Extractor API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_dict(d):
    """Replace NaN values with 0 in a dictionary, safe for lists/arrays."""
    if not isinstance(d, dict):
        return d
    return {k: (0 if (pd.api.types.is_scalar(v) and pd.isna(v)) else v) for k, v in d.items()}

class SaveEditsRequest(BaseModel):
    content_hash: str
    records: List[Dict[str, Any]]

@app.post("/extract-bill")
async def extract_bill(file: UploadFile = File(...)):
    """Extract data from a single medical bill image with full-state caching."""
    try:
        content = await file.read()
        records, metadata = extract_medical_bill_data(content, file.filename)
        
        # If already enriched from cache, return immediately
        if metadata.get('is_enriched'):
            return {
                "data": records,
                "metadata": clean_dict(metadata)
            }
            
        # Otherwise, process and save to enriched cache
        df = parse_and_format_data(records, file.filename)
        df = df.fillna(0)
        data = df.to_dict(orient="records")
        
        # Only cache if we have a content hash (not old file-name based cache)
        if 'content_hash' in metadata:
            save_to_cache(metadata['content_hash'], enriched=data)
            
        return {
            "data": data,
            "metadata": clean_dict(metadata)
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-multiple")
async def extract_multiple(files: List[UploadFile] = File(...)):
    """Extract data from multiple medical bill images with caching."""
    all_records = []
    total_metadata = {"files_processed": 0, "errors": []}
    
    for file in files:
        try:
            content = await file.read()
            records, metadata = extract_medical_bill_data(content, file.filename)
            
            if metadata.get('is_enriched'):
                data = records
            else:
                df = parse_and_format_data(records, file.filename)
                df = df.fillna(0)
                data = df.to_dict(orient="records")
                if 'content_hash' in metadata:
                    save_to_cache(metadata['content_hash'], enriched=data)
            
            all_records.extend(data)
            total_metadata["files_processed"] += 1
        except Exception as e:
            total_metadata["errors"].append({"file": file.filename, "error": str(e)})
            
    return {
        "data": all_records,
        "metadata": total_metadata
    }

@app.post("/highlight-pdf")
async def highlight_pdf(
    file: UploadFile = File(...),
    page: int = Form(...),
    geometry: str = Form(None)
):
    """Generate highlighted image or PDF base64."""
    try:
        content = await file.read()
        geo_dict = json.loads(geometry) if geometry else None
        
        # Return both for flexibility
        img_bytes = render_pdf_page_with_highlight(content, page, geo_dict)
        pdf_base64 = get_pdf_with_highlight_base64(content, page, geo_dict)
        
        return {
            "image": base64.b64encode(img_bytes).decode('utf-8') if img_bytes else None,
            "pdf_base64": pdf_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-batch-pdf")
async def extract_batch_pdf(
    pdf_file: UploadFile = File(...),
    csv_file: UploadFile = File(...),
    json_file: UploadFile = File(...)
):
    """Extract data from a batch PDF using classification CSV and Textract JSON."""
    try:
        pdf_content = await pdf_file.read()
        csv_content = await csv_file.read()
        json_content = await json_file.read()
        
        records, metadata = process_batch_pdf_data(pdf_content, csv_content, json_content)
        
        # If already enriched from cache, return immediately
        if metadata.get('is_enriched'):
            return {
                "data": records,
                "metadata": clean_dict(metadata)
            }
            
        if not records:
            return {
                "data": [],
                "metadata": metadata
            }
            
        # Otherwise, process and enrich
        df = parse_and_format_data(records, pdf_file.filename)
        df = df.fillna(0)
        data = df.to_dict(orient="records")
        
        # Save to enriched cache
        if 'content_hash' in metadata:
            save_to_cache(metadata['content_hash'], enriched=data)
            
        # Check for user edits
        user_data = check_user_cache(metadata['content_hash']) if 'content_hash' in metadata else None
            
        return {
            "data": data,
            "user_data": user_data,
            "metadata": clean_dict(metadata)
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-user-edits")
async def save_user_edits_endpoint(request: SaveEditsRequest):
    """Save user-edited records to cache."""
    try:
        save_user_edits(request.content_hash, request.records)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-demo-data")
async def load_demo_data():
    """Load the demo GP2 data set."""
    try:
        print("Loading demo data...")
        # Define paths
        base_dir = Path.cwd() / "Page_classification"
        print(f"Base dir: {base_dir}")
        
        pdf_path = base_dir / "GP2.pdf"
        csv_path = base_dir / "GP2 Page Classification - Sheet1.csv"
        json_path = base_dir / "b98d314f-2b46-49f3-9986-1dc7249cb449__document_uuid__GP2.json"
        
        print(f"Checking files:\nPDF: {pdf_path}\nCSV: {csv_path}\nJSON: {json_path}")
        
        if not (pdf_path.exists() and csv_path.exists() and json_path.exists()):
            print("Files not found!")
            raise HTTPException(status_code=404, detail=f"Demo files not found at {base_dir}")
            
        # Read files
        print("Reading files...")
        pdf_bytes = pdf_path.read_bytes()
        csv_bytes = csv_path.read_bytes()
        json_bytes = json_path.read_bytes()
        print(f"Read {len(pdf_bytes)} bytes of PDF")
        
        # Reuse the batch processing logic (which handles caching)
        print("Processing batch data...")
        records, metadata = process_batch_pdf_data(pdf_bytes, csv_bytes, json_bytes)
        print("Batch processing complete")
        
        # Prepare response data
        data = []
        user_data = None
        
        # Standardize formatting
        if records:
            print("Formatting records...")
            df = parse_and_format_data(records, "GP2.pdf")
            df = df.fillna(0)
            data = df.to_dict(orient="records")
            
            # Save to enriched cache
            if 'content_hash' in metadata:
                save_to_cache(metadata['content_hash'], enriched=data)
                
            # Check for user edits
            if 'content_hash' in metadata:
                user_data = check_user_cache(metadata['content_hash'])
        
        print("Returning response...")
        return {
            "data": data,
            "user_data": user_data,
            "metadata": clean_dict(metadata),
            "files": {
                "pdf": base64.b64encode(pdf_bytes).decode('utf-8'),
                "pdf_name": "GP2.pdf",
                "csv_name": "GP2 Page Classification.csv",
                "json_name": "GP2 Textract Blocks.json"
            }
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
