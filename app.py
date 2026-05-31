"""FoodLabelSafety Application Entry Point."""

import argparse
import sys
import json
import csv
import subprocess
from pathlib import Path

from src.parser.query_parser import QueryParser
from src.rag.engine import AdditiveRetriever
from src.rag.generator import ResponseGenerator
from src.models.schemas import AnalysisReport

# Paths
ROOT_DIR = Path(__file__).parent
DATA_PATH = ROOT_DIR / "data" / "additives" / "fssai_additives.json"

def print_report(report: AnalysisReport):
    """Format and print the AnalysisReport to the console."""
    print("\n" + "=" * 60)
    print(" [ FoodLabelSafety Analysis Report ] ")
    print("=" * 60)
    
    print("\n[ VERDICT ]")
    
    # Veg status coloring
    veg_icon = "[VEG]" if report.is_vegetarian == "vegetarian" else "[NON-VEG]" if report.is_vegetarian == "non_vegetarian" else "[AMBIGUOUS]"
    print(f"  Dietary Status : {veg_icon} {report.is_vegetarian.upper().replace('_', ' ')}")
    
    # Safety coloring
    safe_icon = "[SAFE]" if "safe" in report.overall_safety else "[WARNING]" if "moderate" in report.overall_safety else "[DANGER]"
    print(f"  Overall Safety : {safe_icon} {report.overall_safety.upper().replace('_', ' ')}")
    
    print("\n[ SUMMARY ]")
    print(f"  {report.summary}")
    print(f"  Guidance: {report.consumption_advice}")
    
    if report.warnings:
        print("\n[ WARNINGS ]")
        for warning in report.warnings:
            print(f"  ! {warning}")
            
    print("\n[ ADDITIVES DETECTED ]")
    for additive in report.additives_found:
        v_icon = "[V]" if additive.veg_status == "vegetarian" else "[NV]" if additive.veg_status == "non_vegetarian" else "[?]"
        s_icon = "[S]" if "safe" in additive.safety_rating else "[W]" if "moderate" in additive.safety_rating else "[D]"
        print(f"  * {additive.code}: {additive.name}")
        print(f"    {v_icon} {additive.veg_status.title()}  |  {s_icon} {additive.safety_rating.title().replace('_', ' ')}")
        print(f"    {additive.brief}")
        print()
    
    print("=" * 60 + "\n")

def check_and_run_setup():
    """Check if the dataset is fully built, and run the setup script if not."""
    try:
        from src.setup.constants import CSV_FILE, OUTPUT_FILE
    except ImportError:
        print("Error: Could not import setup constants. Make sure src/setup exists.")
        sys.exit(1)
        
    needs_setup = False
    
    if not OUTPUT_FILE.exists():
        needs_setup = True
    else:
        # Check if fully built by comparing JSON length to CSV length
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                next(csv_reader) # skip header
                csv_count = sum(1 for row in csv_reader)
                
            # If we are missing entries, we need to resume building
            if len(data) < csv_count:
                print(f"Knowledge base is incomplete ({len(data)}/{csv_count} entries).")
                needs_setup = True
        except Exception:
            needs_setup = True

    if needs_setup:
        print("\n" + "=" * 60)
        print(" 🛠️  Knowledge Base Setup Required ")
        print("=" * 60)
        print("Starting the dataset builder automatically...\n")
        try:
            # Run the setup script as a subprocess
            subprocess.run([sys.executable, "src/setup/build_dataset.py"], check=True)
            print("\nSetup complete! Continuing with application...\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[!] Error running setup script: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="FoodLabelSafety - Analyze food additives from images or text.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="A text query (e.g. 'Is E100 safe?')")
    group.add_argument("--image", type=str, help="Path to a food label image")
    
    args = parser.parse_args()
    
    # 0. Check and run setup if needed
    check_and_run_setup()
    
    # 1. Initialize components
    print("Initializing FoodLabelSafety engine...")
    try:
        from src.setup.constants import OUTPUT_FILE
        data_path = OUTPUT_FILE
    except ImportError:
        data_path = DATA_PATH
        
    if not data_path.exists():
        print(f"Error: Knowledge base not found at {data_path}")
        sys.exit(1)
        
    retriever = AdditiveRetriever(data_path)
    query_parser = QueryParser()
    generator = ResponseGenerator()
    
    # 2. Get input text
    if args.image:
        print("\nLoading OCR Engine...")
        from src.ocr.extractor import LabelExtractor
        extractor = LabelExtractor()
        try:
            input_text = extractor.extract_text(args.image)
            print(f"\nExtracted Text: {input_text[:100]}...\n")
        except Exception as e:
            print(f"OCR Error: {e}")
            sys.exit(1)
    else:
        input_text = args.text
        
    # 3. Parse text for additive codes
    codes = query_parser.extract_codes(input_text)
    if not codes:
        print("\n[!] No valid additive codes (e.g. E100, INS 631) found in the input.")
        print("If you uploaded an image, make sure the ingredients list is clear.")
        sys.exit(0)
        
    print(f"Detected additive codes: {', '.join(codes)}")
    
    # 4. Retrieve knowledge base context
    context = retriever.retrieve(codes)
    if not context:
        print("\n[!] Codes found, but no matching data in the FSSAI knowledge base.")
        sys.exit(0)
        
    print(f"Found {len(context)} matching entries in the database.")
    
    # 5. Generate final report
    report = generator.generate_report(user_query=input_text, additives_context=context)
    if report:
        print_report(report)
    else:
        print("\n[!] Failed to generate report from LLM.")

if __name__ == "__main__":
    main()
