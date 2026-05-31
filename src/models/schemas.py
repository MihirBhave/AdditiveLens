"""Pydantic models for structured data throughout the app."""

from enum import Enum
from pydantic import BaseModel


class VegStatus(str, Enum):
    VEGETARIAN = "vegetarian"
    NON_VEGETARIAN = "non_vegetarian"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class SafetyRating(str, Enum):
    SAFE = "safe"
    GENERALLY_SAFE = "generally_safe"
    MODERATE_CONCERN = "moderate_concern"
    HIGH_CONCERN = "high_concern"
    UNKNOWN = "unknown"


class AdditiveEntry(BaseModel):
    """Schema for a single additive in the knowledge base."""
    ins_code: str
    e_number: str | None = None
    name: str
    category: str
    veg_status: VegStatus
    safety_rating: SafetyRating
    fssai_approved: bool = True
    description: str
    health_concerns: list[str] = []
    common_foods: list[str] = []
    safe_consumption: str = ""
    source: str = ""


class AdditiveAnalysis(BaseModel):
    """Analysis result for a single additive returned by the LLM."""
    code: str
    name: str
    veg_status: VegStatus
    safety_rating: SafetyRating
    brief: str


class AnalysisReport(BaseModel):
    """Complete analysis report returned to the user."""
    additives_found: list[AdditiveAnalysis]
    overall_safety: SafetyRating
    is_vegetarian: VegStatus
    summary: str
    consumption_advice: str
    warnings: list[str] = []
