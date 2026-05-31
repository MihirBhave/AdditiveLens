import ollama
import chromadb
import csv
import re
from pathlib import Path
from constants import *

def chunk_taxonomy(filepath: Path) -> list[dict]:
    """
    Split the Open Food Facts additives.txt into chunks.
    Each additive entry is separated by blank lines.
    
    Returns a list of {"id": "E102", "text": "full block text"} dicts.
    """
    raw = filepath.read_text(encoding="utf-8")
    
    # Split on double newlines (blank lines separate entries)
    blocks = raw.split("\n\n")
    
    chunks = []
    seen_ids = {}  # Track IDs to handle duplicates
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Skip pure comment blocks and synonym/stopword definitions
        lines = block.split("\n")
        non_comment_lines = [l for l in lines if not l.strip().startswith("#")]
        if not non_comment_lines:
            continue
        
        # Skip stopwords and synonyms blocks
        first_meaningful = non_comment_lines[0].strip()
        if first_meaningful.startswith("stopwords:") or first_meaningful.startswith("synonyms:"):
            continue
        
        # Try to extract the E-number for this chunk as an ID
        # Match variants like E101, E101a, E101(i), E101(ii), E150b, etc.
        e_match = re.search(r'\bE\d{3,4}(?:[a-z]|\([ivx]+\))?\b', block, re.IGNORECASE)
        if not e_match:
            continue
        
        chunk_id = e_match.group(0).upper()
        
        # Handle remaining duplicates by appending a counter
        if chunk_id in seen_ids:
            seen_ids[chunk_id] += 1
            chunk_id = f"{chunk_id}_v{seen_ids[chunk_id]}"
        else:
            seen_ids[chunk_id] = 0
        
        chunks.append({
            "id": chunk_id,
            "text": block,
        })
    
    return chunks


def build_chroma_index(chunks: list[dict], force_rebuild: bool = False):
    """
    Embed all taxonomy chunks and store in ChromaDB.
    Uses nomic-embed-text via Ollama for embeddings.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Check if collection already exists
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        if force_rebuild:
            print("  Deleting existing collection for rebuild...")
            client.delete_collection(COLLECTION_NAME)
        else:
            collection = client.get_collection(COLLECTION_NAME)
            count = collection.count()
            if count > 0:
                print(f"  ChromaDB already has {count} entries. Skipping index build.")
                print("  (Use --rebuild-index to force rebuild)")
                return collection
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"  Embedding {len(chunks)} taxonomy chunks...")
    
    # Process in batches (Ollama embedding can handle batches)
    BATCH_SIZE = 20
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        
        # Get embeddings from Ollama
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        
        # Ollama embed endpoint
        embeddings = []
        for text in texts:
            # Truncate to avoid context length limits for embeddings
            safe_text = text[:4000]
            resp = ollama.embed(model=EMBEDDING_MODEL, input=safe_text)
            embeddings.append(resp["embeddings"][0])
        
        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
        )
        
        done = min(i + BATCH_SIZE, len(chunks))
        print(f"[!] Indexed {done}/{len(chunks)} chunks", end="\r")
    
    print(f"\n [+] Indexed {len(chunks)} chunks into ChromaDB")
    return collection


def read_csv(filepath: Path) -> list[dict]:
    """Read the SuhasDissa additives CSV."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "e_code": row["e_code"].strip(),
                "title": row["title"].strip(),
                "info": row["info"].strip(),
                "category": row["e_type"].strip(),
                "halal_status": row.get("halal_status", "").strip(),
            })
    return rows


def retrieve_taxonomy_context(collection, e_code: str, title: str, top_k: int = 2) -> str:
    """
    Search ChromaDB for the taxonomy entry matching this additive.
    Returns the retrieved text context.
    """
    e_code_upper = e_code.upper()
    
    # Fast path: Exact ID match
    results = collection.get(ids=[e_code_upper])
    context = ""
    
    if results and results["documents"] and len(results["documents"]) > 0:
        context = results["documents"][0]
    else:
        # Fallback: Semantic search
        query = f"{e_code} {title}"
        resp = ollama.embed(model=EMBEDDING_MODEL, input=query)
        query_embedding = resp["embeddings"][0]
        
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
        )
        if search_results and search_results["documents"] and search_results["documents"][0]:
            context = search_results["documents"][0][0]
            
    if not context:
        return ""
        
    # OPTIMIZATION: Strip out non-English translations to speed up LLM prompt processing
    cleaned_lines = []
    for line in context.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Keep english names, vegetarian/vegan tags, classes, and efsa tags
        if line.startswith("en:") or "en:" in line or "efsa" in line:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)