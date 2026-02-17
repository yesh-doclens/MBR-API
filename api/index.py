"""
Vercel serverless entry point for the Medical Bill Extractor API.
This file imports the FastAPI app and makes it compatible with Vercel's serverless runtime.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import from app module
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

# Vercel will use this app instance
# The app is already configured with all routes and middleware in app/main.py
