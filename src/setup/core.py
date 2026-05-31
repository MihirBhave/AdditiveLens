import json
import ollama
import re

from constants import *
from utils import retrieve_taxonomy_context

def enrich_entry(collection, csv_row: dict) -> dict | None:
    """
    For one CSV row:
      1. Retrieve matching taxonomy context from ChromaDB
      2. Send to LLM with enrichment prompt
      3. Parse and return structured JSON
    """
    e_code = csv_row["e_code"]
    title = csv_row["title"]
    category = csv_row["category"]
    description = csv_row["info"]
    ins_code = re.sub(r'^[Ee]', '', e_code)
    
    # Retrieve Context
    context = retrieve_taxonomy_context(collection, e_code, title)
    if not context:
        context = "(No matching reference data found in taxonomy)"
    
    # Find the vegetarian status from the context(chunk of data corresponding to each ecode)
    veg_status_val = "unknown"
    for line in context.split("\n"):
        if "vegetarian:en:" in line:
            val = line.split(":")[-1].strip().lower()
            if val == "yes": veg_status_val = "vegetarian"
            elif val == "no": veg_status_val = "non_vegetarian"
            elif val == "maybe": veg_status_val = "ambiguous"
            break
            
    # Truncate description to 600 chars
    desc_truncated = description[:600] if len(description) > 600 else description
    
    # Build the final prompt
    prompt = ENRICHMENT_PROMPT.format(
        context=context,
        title=title,
        e_code=e_code,
        category=category,
        description=desc_truncated,
        ins_code=ins_code,
    )
    
    # Call LLM using chat option
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,       # High Determinism and low creativity
                "num_predict": 512,        # Limit output length
            },
        )
        
        content = response["message"]["content"].strip()
        
        # Extract JSON from response (handle markdown fences if model adds them)
        # Try to find JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            
            # Ensure required fields exist with fallbacks
            result.setdefault("ins_code", ins_code)
            result.setdefault("e_number", e_code)
            result.setdefault("name", title)
            result.setdefault("category", category)
            result.setdefault("veg_status", "unknown")
            result.setdefault("safety_rating", "unknown")
            result.setdefault("description", desc_truncated[:200])
            result.setdefault("source", "")
            result.setdefault("health_concerns", [])
            result.setdefault("safe_consumption", "")
            result.setdefault("common_foods", [])
            result.setdefault("fssai_approved", True)
            
            # Override LLM's veg_status with our 100% accurate programmatic extraction
            result["veg_status"] = veg_status_val
            
            # Validate enum values
            valid_safety = {"safe", "generally_safe", "moderate_concern", "high_concern", "unknown"}
            if result["safety_rating"] not in valid_safety:
                result["safety_rating"] = "unknown"
            
            return result
        else:
            print(f"[-] Could not extract JSON from LLM response for {e_code}")
            print(f"[+] Raw response: {content[:200]}...")
            return None
            
    except json.JSONDecodeError as e:
        print(f"[-] JSON parse error for {e_code}: {e}")
        return None
    except Exception as e:
        print(f"[-] LLM call failed for {e_code}: {e}")
        return None


def load_progress() -> dict:
    """Load already-processed entries from the progress file."""
    
    completed = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        completed[entry["e_number"]] = entry
                    except json.JSONDecodeError:
                        continue
    return completed


def save_progress(entry: dict):
    """Append one completed entry to the progress file."""
    
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compile_final_output(completed: dict):
    """Compile all completed entries into the final JSON file."""
    # Sort by INS code numerically
    entries = list(completed.values())
    
    def sort_key(e):
        code = re.sub(r'[^0-9]', '', e.get("ins_code", "9999"))
        return int(code) if code else 9999
    
    entries.sort(key=sort_key)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    
    print(f"\n[+] Saved {len(entries)} entries to {OUTPUT_FILE}")