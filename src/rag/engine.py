"""In-memory retrieval engine for the knowledge base."""

import json
from pathlib import Path
from typing import Optional

from src.models.schemas import AdditiveEntry

class AdditiveRetriever:
    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)
        self.data: list[AdditiveEntry] = []
        
        # Fast lookup tables
        self.by_e_code: dict[str, AdditiveEntry] = {}
        self.by_ins_code: dict[str, AdditiveEntry] = {}
        
        self._load_data()

    def _load_data(self):
        """Load JSON into memory and build lookup tables."""
        if not self.json_path.exists():
            print(f"Warning: Knowledge base not found at {self.json_path}")
            return
            
        with open(self.json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        for item in raw_data:
            entry = AdditiveEntry(**item)
            self.data.append(entry)
            
            # Map E-numbers (e.g. "e100", "e101(i)")
            if entry.e_number:
                self.by_e_code[entry.e_number.lower()] = entry
                
            # Map INS-numbers (e.g. "100", "101(i)")
            self.by_ins_code[entry.ins_code.lower()] = entry

    def retrieve(self, codes: list[str]) -> list[AdditiveEntry]:
        """Fetch full additive JSON data for a list of standardized codes."""
        results = []
        seen = set()
        
        for code in codes:
            code_lower = code.lower()
            
            if code_lower in seen:
                continue
                
            # Try E-code match
            if code_lower in self.by_e_code:
                results.append(self.by_e_code[code_lower])
                seen.add(code_lower)
                continue
                
            # Try INS-code match (strip the leading 'e')
            ins = code_lower.replace('e', '').strip()
            if ins in self.by_ins_code:
                results.append(self.by_ins_code[ins])
                seen.add(code_lower)
                continue
                
        return results
