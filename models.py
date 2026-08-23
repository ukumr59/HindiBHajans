"""
Core data structures for the discovery/scoring layer.

A StoryCandidate is the atomic unit the whole pipeline works with.
Everything downstream (script writer, video assembler, publisher)
consumes a ranked list of these.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class StoryCategory(str, Enum):
    """Coarse category used to pick a visual template later in the pipeline."""
    DEFENCE_DEAL = "defence_deal"
    DIPLOMACY_BILATERAL = "diplomacy_bilateral"
    BORDER_SECURITY = "border_security"
    TRADE_ENERGY = "trade_energy"
    STRATEGIC_TECH = "strategic_tech"
    MULTILATERAL = "multilateral"          # UN, QUAD, BRICS, SCO, G20 etc.
    INTERNATIONAL_MENTION = "international_mention"  # India mentioned, not central
    NOT_RELEVANT = "not_relevant"


@dataclass
class StoryCandidate:
    headline: str
    summary: str                 # 2-4 sentence factual summary, in our own words
    source_name: str
    source_url: str
    published_at: datetime
    discovered_at: datetime = field(default_factory=datetime.utcnow)

    # Filled in by the scorer:
    category: StoryCategory | None = None
    india_relevance: float = 0.0        # 0-10 : is India the actual subject?
    strategic_importance: float = 0.0   # 0-10 : does this matter geopolitically?
    freshness: float = 0.0              # 0-10 : recency decay
    virality_signal: float = 0.0        # 0-10 : momentum/discussion potential
    corroboration: float = 0.0          # 0-10 : how many independent sources carry it
    hook_strength: float = 0.0          # 0-10 : is there a clean narrative hook?
    final_score: float = 0.0
    reasoning: str = ""                 # human-readable justification
    rejected: bool = False
    rejection_reason: str = ""
