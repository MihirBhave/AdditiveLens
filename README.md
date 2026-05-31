# AdditiveLens 

An offline-first, AI-powered food label analyzer that scans ingredient lists and warns you about hazardous additives, dietary status, and hyper-processed chemicals.

Built with **Python**, **EasyOCR**, and **Ollama**, this project bridges Computer Vision with a custom Local LLM pipeline to demystify complex food labels (like E-numbers and INS codes) using the Open Food Facts taxonomy.

---

## Features

- **Image OCR Scanning:** Snap a picture of a food label. The app uses OpenCV and EasyOCR to extract the ingredients text.
- **Smart Additive Parsing:** A custom two-pass regex engine extracts the additive codes (like `471` or `102`) by analyzing their proximity to food science keywords (like "emulsifier" or "stabilizer"), helping to identify labels without explicit E or INS prefixes to the codes.
- **Exact-Match RAG Engine:** Bypasses slow vector databases (ChromaDB) in favor of a lightning-fast $O(1)$ in-memory JSON retriever for structured taxonomy data.
- **Fully Local & Private:** Runs entirely offline using `phi4-mini:3.8b-q4_K_M` model via Ollama.
- **Structured Reports:** Leverages Pydantic schemas to force the LLM to output a perfectly structured JSON safety report, flagging carcinogens, allergens, and ambiguous vegetarian statuses.

---

## Architecture & Pipeline

This project employs a unique **Structured RAG (Retrieval-Augmented Generation)** architecture tailored for precise identifier lookups rather than fuzzy semantic search.

1. **Dataset Builder (`src/setup/build_dataset.py`)**
   - Parses the raw `additives.txt` taxonomy from Open Food Facts.
   - Uses an LLM to enrich and structure the data into a clean, 500+ item JSON dictionary mapped by E-number.
   - This script is invoked only the JSON dictionary is missing.
   - Most GPU intensive process as we use semantic-search RAG to generate the JSON output from the knowledge base and given csv row.
2. **Text Extraction (`src/ocr/extractor.py`)**
   - Applies OpenCV grayscale and contrast filters to the image.
   - EasyOCR reads the text blocks.
3. **Query Parsing (`src/parser/query_parser.py`)**
   - Identifies E-numbers and INS codes from the noisy OCR output.
4. **Retrieval (`src/rag/engine.py`)**
   - Fetches the exact JSON profile for every detected additive from the local knowledge base.
5. **Generation (`src/rag/generator.py`)**
   - Injects the JSON context into the Ollama prompt.
   - The LLM acts as an FSSAI food inspector, analyzing the additives and returning a `AnalysisReport` Pydantic object.

---

## Tech Stack

- **Language:** Python 3.12+
- **Computer Vision:** OpenCV, EasyOCR
- **AI / LLM:** Ollama (`phi4-mini:3.8b-q4_K_M`), Pydantic (Structured Outputs)
- **Data Source:** Open Food Facts (Taxonomy)

---

## Setup & Installation

1. **Clone the repository**
2. **Install Ollama** and pull the required models:
   ```bash
   ollama pull phi4-mini:3.8b-q4_K_M
   ollama pull nomic-embed-text
   ```
3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Build the Knowledge Base(GPU Intensive Process):**
   *(Note: The app will automatically run this on first boot if missing)*
   ```bash
   python src/setup/build_dataset.py
   ```

---

## Usage

**Analyze an Image:**
```bash
python app.py --image "<Path_To_Your_Image>"
```

**Analyze a Text Query:**
```bash
python app.py --text "Are E102 and E407 safe?"
```

### Sample Output
```text
============================================================
 [ FoodLabelSafety Analysis Report ] 
============================================================

[ VERDICT ]
  Dietary Status : [AMBIGUOUS] AMBIGUOUS
  Overall Safety : [WARNING] MODERATE CONCERN

[ SUMMARY ]
  The product contains several additives with varying safety ratings. Carrageenan poses a high concern due to potential toxic risks...
```

## Future Work
   - This project can be upgraded to work with a better text extraction model.
   - Can be expanded into a full-fledged application with a web-interface. 
   - Can adapt the Knowledge Base to completely fit to Indian needs. For example, the veg_status is mostly ambigious as most of the additives are created using non-vegetarian sources outside India due to which veg_status is mostly logged as "ambigious" by the model. 

---

## Credits & Acknowledgements

This project would not be possible without the open-source data provided by the following initiatives:
- **[Open Food Facts](https://world.openfoodfacts.org/)**: For providing the incredibly detailed, community-driven global taxonomy of food additives (`additives.txt`).
- **[Suhas Dissa's FSSAI Additives Dataset](https://github.com/suhasdissa)**: For providing the base CSV mapping of E-numbers to their FSSAI regulatory status in India.
