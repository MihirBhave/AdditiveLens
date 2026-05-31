"""LLM generation engine using Ollama and Pydantic structured output."""

import json
from typing import Optional

import ollama
from src.models.schemas import AnalysisReport, AdditiveEntry

SYSTEM_PROMPT = """You are a food safety expert specializing in Indian food regulations (FSSAI).
The user wants to know about the safety and vegetarian status of additives found in their food label or query.
I will provide you with the user's original query/label text, and the EXACT JSON profiles of the additives found.

YOUR TASK:
1. Read the provided JSON profiles carefully.
2. For EACH additive found in the JSON context, populate the 'additives_found' list with its code, name, veg_status, safety_rating, and a very brief 1-sentence summary.
3. Provide an 'overall_safety' rating for the product based on the worst additive found.
4. Provide an 'is_vegetarian' status (if any additive is non_vegetarian, the whole product is non_vegetarian. If any is ambiguous, it is ambiguous).
5. Write a crisp 2-3 sentence 'summary' answering the user's core concern.
6. Provide general 'consumption_advice' (e.g. "Safe for daily consumption" or "Consume occasionally due to X").
7. Add any specific health warnings to the 'warnings' list.

CRITICAL RULES:
- DO NOT hallucinate. Use ONLY the data provided in the KNOWLEDGE BASE CONTEXT JSON.
- DO NOT invent or extract new additives from the USER QUERY text (e.g., if the user text has random numbers like 1290, ignore them unless they are in the JSON context).
- Keep the summary short, crisp, and direct.
"""

class ResponseGenerator:
    def __init__(self, model_name: str = "phi4-mini:3.8b-q4_K_M"):
        self.model = model_name

    def generate_report(self, user_query: str, additives_context: list[AdditiveEntry]) -> Optional[AnalysisReport]:
        """
        Generate a structured AnalysisReport from the LLM based on context.
        """
        if not additives_context:
            return None
            
        # Format the context into a string representation
        context_json = [entry.model_dump(mode='json') for entry in additives_context]
        context_str = json.dumps(context_json, indent=2)
        
        # Prevent hallucination on long OCR dumps by hiding the raw text from the LLM
        if len(user_query) > 200:
            user_message = f"KNOWLEDGE BASE CONTEXT:\n{context_str}\n\nTASK: The user provided a food label image. Please summarize the safety of the additives provided in the context."
        else:
            user_message = f"USER QUERY:\n{user_query}\n\nKNOWLEDGE BASE CONTEXT:\n{context_str}"
        
        print(f"Generating summary with {self.model}...")
        
        try:
            # We use Ollama's native structured output feature by passing the Pydantic model
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                format=AnalysisReport.model_json_schema(),
                options={"temperature": 0.1}  # Low temp for factual consistency
            )
            
            # The response is a JSON string matching the schema
            json_str = response['message']['content']
            report = AnalysisReport.model_validate_json(json_str)
            return report
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
