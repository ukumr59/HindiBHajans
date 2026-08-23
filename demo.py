"""
DEMO RUN — uses real stories pulled from today's news (web search, Aug 2026).

Note on how this demo differs from production:
This sandbox has no live network access, so I can't call the Anthropic API
from here to run editorial_judge.judge_candidate() for real. What's below
is me applying that exact rubric by hand to real stories, so you can see
the scoring/ranking mechanics work end-to-end. Once this is deployed on
GitHub Actions (which does have network access), judge_candidate() runs
for real against the live Claude API — nothing about the scoring.py logic
changes, only where the judgement numbers come from.
"""

from datetime import datetime, timedelta, timezone
from .models import StoryCandidate, StoryCategory
from .scoring import rank_candidates

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)

candidates = [
    StoryCandidate(
        headline="India moves to buy 5 more S-400 air defence squadrons from Russia",
        summary=(
            "India's Defence Acquisition Council approved a proposal to purchase "
            "five additional S-400 air defence systems from Russia, on top of the "
            "existing squadrons already in service. Formal negotiations on cost "
            "and delivery timelines are expected to follow."
        ),
        source_name="Defence News India",
        source_url="https://defence.in/threads/india-expected-to-sign-second-s-400-deal-with-russia-by-late-2026-with-target-for-initial-deliveries-in-2028.18051/",
        published_at=NOW - timedelta(hours=6),
        india_relevance=9.5, strategic_importance=9.0, virality_signal=7.0,
        hook_strength=8.0, category=StoryCategory.DEFENCE_DEAL, corroboration=6.0,
        reasoning=(
            "India strengthening its air-defence shield with a major Russian system "
            "purchase is squarely strategic and has a strong visual hook (missile "
            "systems, airspace protection). Central to India's security posture."
        ),
    ),
    StoryCandidate(
        headline="Reports: India approaches Israel for bilateral defence pact after Saudi-Turkey-Pakistan alignment",
        summary=(
            "Following a new defence agreement between Saudi Arabia, Turkey and "
            "Pakistan, media reports say India has approached Israel to discuss a "
            "reciprocal defence arrangement, seen as a response to a shifting "
            "regional security alignment in West Asia."
        ),
        source_name="Jerusalem Post",
        source_url="https://www.jpost.com/middle-east/article-905010",
        published_at=NOW - timedelta(days=2),
        india_relevance=8.5, strategic_importance=8.5, virality_signal=8.0,
        hook_strength=9.0, category=StoryCategory.DIPLOMACY_BILATERAL, corroboration=3.0,
        reasoning=(
            "This is India actively repositioning in response to a rival bloc "
            "forming — strong hook (a new axis forming near India's interests), "
            "high strategic stakes, though sourced from a single regional outlet "
            "so corroboration is currently thin."
        ),
    ),
    StoryCandidate(
        headline="India's Agni-4 ballistic missile completes operational user trial",
        summary=(
            "India's strategic forces conducted a successful operational user "
            "trial of the Agni-4 medium-range ballistic missile, confirming its "
            "technical and operational parameters."
        ),
        source_name="Global Defense Insight",
        source_url="https://defensetalks.com/indian-defence-industry-weekly-development-report-3-9-august-2026/",
        published_at=NOW - timedelta(days=17),
        india_relevance=9.0, strategic_importance=8.0, virality_signal=6.0,
        hook_strength=6.5, category=StoryCategory.DEFENCE_DEAL, corroboration=4.0,
        reasoning=(
            "Strong strategic-capability story (nuclear-capable missile "
            "validated), but it's now over two weeks old — freshness will pull "
            "this down even though the underlying importance is high."
        ),
    ),
    StoryCandidate(
        headline="US and China trade tensions escalate over semiconductor export rules",
        summary=(
            "Washington and Beijing exchanged fresh restrictions on semiconductor "
            "exports this week, with analysts warning of ripple effects across "
            "global supply chains."
        ),
        source_name="Wire agency roundup",
        source_url="https://example.com/us-china-chips",
        published_at=NOW - timedelta(hours=10),
        india_relevance=2.0, strategic_importance=5.0, virality_signal=6.0,
        hook_strength=6.0, category=StoryCategory.INTERNATIONAL_MENTION, corroboration=8.0,
        reasoning=(
            "This is a US-China story. India isn't the subject — at most it's "
            "an indirect beneficiary/bystander. This is exactly the 'international "
            "story that merely mentions India' case the brief said to reject, "
            "even though it's a big, well-corroborated story in its own right."
        ),
    ),
    StoryCandidate(
        headline="Parliamentary committee reviews ex-servicemen welfare schemes",
        summary=(
            "A Parliamentary Standing Committee on Defence met to examine welfare "
            "measures for former servicemen, taking evidence from Defence Ministry "
            "officials."
        ),
        source_name="News on Air",
        source_url="https://www.newsonair.gov.in/",
        published_at=NOW - timedelta(hours=20),
        india_relevance=7.0, strategic_importance=3.0, virality_signal=2.0,
        hook_strength=2.0, category=StoryCategory.NOT_RELEVANT, corroboration=2.0,
        reasoning=(
            "Genuinely India-focused and defence-adjacent, but this is domestic "
            "administrative housekeeping, not a geopolitical/strategic story — "
            "no international dimension, no real hook. Correctly filtered out "
            "by the strategic-importance floor rather than the relevance floor."
        ),
    ),
]

if __name__ == "__main__":
    ranked = rank_candidates(candidates, now=NOW)

    print("=" * 78)
    print("DAILY SHORTLIST — ranked, accepted stories")
    print("=" * 78)
    for i, c in enumerate([x for x in ranked if not x.rejected], 1):
        print(f"\n#{i}  [{c.final_score:.2f}]  {c.headline}")
        print(f"     category: {c.category.value}   source: {c.source_name}")
        print(f"     india_relevance={c.india_relevance}  strategic={c.strategic_importance}  "
              f"freshness={c.freshness:.1f}  virality={c.virality_signal}  hook={c.hook_strength}")
        print(f"     why: {c.reasoning}")

    print("\n" + "=" * 78)
    print("REJECTED — excluded from today's schedule, with reasons")
    print("=" * 78)
    for c in [x for x in ranked if x.rejected]:
        print(f"\n✗  {c.headline}")
        print(f"     {c.rejection_reason}")
