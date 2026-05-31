"""
This script uses the 'additives.txt' as the RAG knowledge base as it is semi0s

Pipeline:
  1. Load additives.txt → chunk by additive entry → embed → store in ChromaDB
  2. Read each row from additives.csv
  3. Query ChromaDB to find the matching taxonomy entry
  4. Send CSV row + retrieved context to phi4-mini → get structured JSON fields
  5. Save enriched data to fssai_additives.json (incremental, resumable)

Prerequisites:
  - Ollama running locally with phi4-mini and nomic-embed-text pulled
  - Raw data files in data/raw/ (run download_data.ps1 first)
  - pip install ollama chromadb

Usage:
  python scripts/build_dataset.py                  # full pipeline
  python scripts/build_dataset.py --rebuild-index   # rebuild ChromaDB from scratch
  python scripts/build_dataset.py --start-from 50   # resume from entry 50
  python scripts/build_dataset.py --dry-run         # test with first 5 entries only
"""

import re
import sys
import time
import argparse
from constants import *
from core import enrich_entry, load_progress, save_progress, compile_final_output
import ollama
from utils import chunk_taxonomy, build_chroma_index, read_csv, retrieve_taxonomy_context


# ─────────────────────────────────────────────
# Step 1: Chunk and index additives.txt
# ─────────────────────────────────────────────



def main():
    parser = argparse.ArgumentParser(description="Build FoodLabelSafety knowledge base")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force rebuild the ChromaDB index from taxonomy")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip the first N entries (for manual resumption)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only the first 5 entries for testing")
    parser.add_argument("--compile-only", action="store_true",
                        help="Just compile progress.jsonl into fssai_additives.json")
    args = parser.parse_args()
    
    # ── Compile-only mode ──
    if args.compile_only:
        completed = load_progress()
        if completed:
            compile_final_output(completed)
        else:
            print("No progress file found. Run the full pipeline first.")
        return
    
    # ── Verify prerequisites ──
    print("=" * 60)
    print("  FoodLabelSafety — Knowledge Base Builder")
    print("=" * 60)
    
    if not TAXONOMY_FILE.exists() or not CSV_FILE.exists():
        print("\n[!] Raw dataset files missing. Downloading automatically...")
        from src.setup.download_data import download_all
        download_all()
    
    # Quick Ollama health check
    print("\n[1/5] Checking Ollama models...")
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        
        llm_found = any(LLM_MODEL in name for name in model_names)
        embed_found = any(EMBEDDING_MODEL in name for name in model_names)
        
        if not llm_found:
            print(f"  [X] Model '{LLM_MODEL}' not found. Run: ollama pull {LLM_MODEL}")
            sys.exit(1)
        if not embed_found:
            print(f"  [X] Model '{EMBEDDING_MODEL}' not found. Run: ollama pull {EMBEDDING_MODEL}")
            sys.exit(1)
        
        print(f"  [OK] {LLM_MODEL} — ready")
        print(f"  [OK] {EMBEDDING_MODEL} — ready")
    except Exception as e:
        print(f"  [X] Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running (ollama serve)")
        sys.exit(1)
    
    # ── Step 1: Chunk and index the taxonomy ──
    print("\n[2/5] Loading Open Food Facts taxonomy...")
    chunks = chunk_taxonomy(TAXONOMY_FILE)
    print(f"  Parsed {len(chunks)} additive entries from taxonomy")
    
    print("\n[3/5] Building ChromaDB index...")
    collection = build_chroma_index(chunks, force_rebuild=args.rebuild_index)
    
    # ── Step 2: Read CSV ──
    print("\n[4/5] Reading SuhasDissa CSV...")
    csv_rows = read_csv(CSV_FILE)
    print(f"  Found {len(csv_rows)} additive entries")
    
    # ── Load previous progress ──
    completed = load_progress()
    if completed:
        print(f"  Resuming: {len(completed)} entries already processed")
    
    # ── Step 3-4: Enrich each entry ──
    if args.dry_run:
        csv_rows = csv_rows[:5]
        print("\n  [TEST] DRY RUN: Processing only 5 entries")
    
    total = len(csv_rows)
    skipped = 0
    errors = 0
    newly_processed = 0
    
    print(f"\n[5/5] Enriching entries with {LLM_MODEL}...")
    print(f"  Estimated time: ~{(total - len(completed)) * 18 // 60} minutes on CPU")
    print("-" * 60)
    
    start_time = time.time()
    
    for i, row in enumerate(csv_rows):
        # Skip if before start-from
        if i < args.start_from:
            continue
        
        e_code = row["e_code"]
        
        # Skip if already processed
        if e_code in completed:
            skipped += 1
            continue
        
        # Process
        elapsed = time.time() - start_time
        rate = newly_processed / elapsed if elapsed > 0 and newly_processed > 0 else 0
        remaining = (total - i) / rate / 60 if rate > 0 else 0
        
        print(f"  [{i+1}/{total}] {e_code} — {row['title'][:40]}", end="")
        if remaining > 0:
            print(f"  (ETA: {remaining:.0f} min)", end="")
        print("...", flush=True)
        
        result = enrich_entry(collection, row)
        
        if result:
            completed[e_code] = result
            save_progress(result)
            newly_processed += 1
            print(f"           -> veg: {result['veg_status']}, safety: {result['safety_rating']}")
        else:
            errors += 1
            # Save a minimal fallback entry so we don't retry forever
            fallback = {
                "ins_code": re.sub(r'^[Ee]', '', e_code),
                "e_number": e_code,
                "name": row["title"],
                "category": row["category"],
                "veg_status": "unknown",
                "safety_rating": "unknown",
                "description": row["info"][:200],
                "source": "",
                "health_concerns": [],
                "safe_consumption": "",
                "common_foods": [],
                "fssai_approved": True,
            }
            completed[e_code] = fallback
            save_progress(fallback)
        
        # Small delay to let CPU breathe
        time.sleep(0.3)
    
    # ── Compile final output ──
    compile_final_output(completed)
    
    # ── Summary ──
    elapsed_total = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" [+] DONE in {elapsed_total / 60:.1f} minutes")
    print(f" [!] Processed: {newly_processed} new entries")
    print(f" [!] Skipped:   {skipped} (already done)")
    print(f" [!] Errors:    {errors} (saved with fallback data)")
    print(f" [+] Total:     {len(completed)} entries in {OUTPUT_FILE.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
