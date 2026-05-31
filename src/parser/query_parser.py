"""Parser for extracting additive codes from text."""

import re

class QueryParser:
    def __init__(self):
        # Explicit codes: E100, INS 100, INS-100
        self.explicit_pattern = re.compile(
            r'\b(?:E|INS)[\s-]*(?P<code>\d{3,4}[a-zA-Z]?(?:\([a-ziv]+\))?)\b',
            re.IGNORECASE
        )
        
        # Keywords that indicate additives
        self.keywords = [
            "colour", "color", "preservative", "emulsifier", "stabilizer", "stabiliser",
            "antioxidant", "flavour", "flavor", "regulator", "agent", "sweetener",
            "firming", "foaming", "glazing", "humectant", "thickener", "raising",
            "additive"
        ]

    def _standardize(self, code: str) -> str:
        """Clean and prefix the code."""
        code = code.strip().lower()
        code = re.sub(r'\s+\(', '(', code)
        return f"E{code}"

    def extract_codes(self, text: str) -> list[str]:
        """
        Extract all unique additive codes from a block of text.
        """
        extracted = set()
        
        # 1. Standard explicit codes (E100, INS 471)
        for match in self.explicit_pattern.finditer(text):
            extracted.add(self._standardize(match.group('code')))
            
        # 2. Short queries (if user just types "471" or "100")
        if len(text.strip()) < 15:
            for match in re.finditer(r'\b(?P<code>\d{3,4}[a-zA-Z]?)\b', text):
                extracted.add(self._standardize(match.group('code')))
                
        # 3. Naked codes near keywords (e.g., "stabilizers (407, 466, 415, 412)")
        for kw in self.keywords:
            for match in re.finditer(rf'\b{kw}\b', text, re.IGNORECASE):
                # Look at the next 60 characters following the keyword
                window = text[match.end() : match.end() + 60]
                
                # Extract any 3-4 digit numbers from this window
                num_matches = re.finditer(r'\b(?P<code>\d{3,4}[a-zA-Z]?)\b', window)
                for n_match in num_matches:
                    extracted.add(self._standardize(n_match.group('code')))
                    
        return list(extracted)
