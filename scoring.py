"""
Deterministic scoring math.

Split deliberately from editorial_judge.py:
  - THIS file does pure math (freshness decay, weighted sum). No LLM, no
    network, fully unit-testable, cheap to run on every candidate.
  - editorial_judge.py does the judgement calls (is India central? how
    strategically important? what's the hook?) that genuinely need an LLM,
    not a keyword rule.

Keeping them separate means the expensive/slow LLM call only ever runs
once per candidate, and the ranking weights can be tuned without touching
the judgement logic at all.
"""

from datetime import datetime, timezone
from .models import StoryCandidate

# Tunable weights. These are exactly what you'd adjust after a few weeks
# of watching what the channel's audience actually engages with.
WEIGHTS = {
    "india_relevance": 0.30,
    "strategic_importance": 0.25,
    "freshness": 0.15,
    "virality_signal": 0.15,
    "corroboration": 0.05,
    "hook_strength": 0.10,
}

# A story below this relevance floor is auto-rejected regardless of
# how strong it is on other axes. This is what stops "international
# story that merely mentions India" from ever reaching the shortlist.
MIN_INDIA_RELEVANCE = 6.0
MIN_STRATEGIC_IMPORTANCE = 4.0


def freshness_score(published_at: datetime, now: datetime | None = None) -> float:
    """Simple decay: full marks inside 12h, tapering to near-zero by 72h."""
    now = now or datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - published_at).total_seconds() / 3600)
    if age_hours <= 12:
        return 10.0
    if age_hours >= 72:
        return 1.0
    # linear taper between 12h and 72h
    return 10.0 - 9.0 * (age_hours - 12) / (72 - 12)


def score_candidate(candidate: StoryCandidate, now: datetime | None = None) -> StoryCandidate:
    """
    Assumes india_relevance, strategic_importance, virality_signal,
    corroboration, hook_strength, and category have already been set
    (by the editorial judge step). Computes freshness + final weighted score,
    and applies the hard rejection floors.
    """
    candidate.freshness = freshness_score(candidate.published_at, now)

    if candidate.india_relevance < MIN_INDIA_RELEVANCE:
        candidate.rejected = True
        candidate.rejection_reason = (
            f"India relevance {candidate.india_relevance:.1f}/10 is below the "
            f"floor of {MIN_INDIA_RELEVANCE} — India is not genuinely central "
            f"to this story, so it's excluded rather than forced into the schedule."
        )
        candidate.final_score = 0.0
        return candidate

    if candidate.strategic_importance < MIN_STRATEGIC_IMPORTANCE:
        candidate.rejected = True
        candidate.rejection_reason = (
            f"Strategic importance {candidate.strategic_importance:.1f}/10 is "
            f"below the floor of {MIN_STRATEGIC_IMPORTANCE} — India-relevant but "
            f"not a geopolitically significant enough story for this channel."
        )
        candidate.final_score = 0.0
        return candidate

    candidate.final_score = sum(
        getattr(candidate, factor) * weight for factor, weight in WEIGHTS.items()
    )
    return candidate


def rank_candidates(candidates: list[StoryCandidate], now: datetime | None = None) -> list[StoryCandidate]:
    scored = [score_candidate(c, now) for c in candidates]
    accepted = sorted((c for c in scored if not c.rejected), key=lambda c: c.final_score, reverse=True)
    rejected = [c for c in scored if c.rejected]
    return accepted + rejected
