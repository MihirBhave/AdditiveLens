from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "additives"
CHROMA_DIR = DATA_DIR / "chroma_dataset_builder"

# Files
TAXONOMY_FILE = RAW_DIR / "additives.txt"
CSV_FILE = RAW_DIR / "additives.csv"
OUTPUT_FILE = OUTPUT_DIR / "fssai_additives.json"
PROGRESS_FILE = OUTPUT_DIR / "enrichment_progress.jsonl"  # Keep appending to this file to save intermediate results

# Ollama models
LLM_MODEL = "phi4-mini:3.8b-q4_K_M"
EMBEDDING_MODEL = "nomic-embed-text"

# ChromaDB collection name
COLLECTION_NAME = "off_additives_taxonomy"

# Enrichment Prompt 
ENRICHMENT_PROMPT = """You are a food safety data expert. I need you to create a structured JSON record for a food additive by combining two data sources.

SOURCE 1 - REFERENCE DATA (from Open Food Facts taxonomy):
{context}

SOURCE 2 - EXISTING DESCRIPTION:
Name: {title}
E-Code: {e_code}
Category: {category}
Description: {description}

TASK: Using BOTH sources, produce a single JSON object with these exact keys. Use the reference data for vegetarian/vegan status (look for "vegetarian:en:" and "vegan:en:" lines). Use the description for health concerns and safety assessment.

{{
  "ins_code": "{ins_code}",
  "e_number": "{e_code}",
  "name": "{title}",
  "category": "{category}",
  "veg_status": "vegetarian | non_vegetarian | ambiguous | unknown",
  "safety_rating": "safe | generally_safe | moderate_concern | high_concern",
  "description": "A concise 1-2 sentence description of what this additive is",
  "source": "Origin of the additive (e.g. Synthetic, Plant-derived, Animal-derived, Mineral)",
  "health_concerns": ["concern1", "concern2"],
  "safe_consumption": "1-2 sentence consumption guidance",
  "common_foods": ["food1", "food2", "food3"],
  "fssai_approved": true
}}

RULES:
- For veg_status: If reference data says "vegetarian:en: yes" → "vegetarian". If "no" → "non_vegetarian". If "maybe" → "ambiguous". If not found → infer from the description, or "unknown".
- For safety_rating: "safe" = no known issues. "generally_safe" = minor concerns for some people. "moderate_concern" = linked to health risks in studies. "high_concern" = banned in some countries or linked to serious health issues.
- For fssai_approved: Set false ONLY if the description explicitly says it's banned or prohibited in India/EU.
- Keep all text fields concise.
- Respond with ONLY the JSON object. No explanations, no markdown fences."""