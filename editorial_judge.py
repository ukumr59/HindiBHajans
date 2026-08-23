"""
This is the module I meant when I said "build the judgement layer first."

Everything else in the pipeline is mechanical. This is the one place that
has to actually think: is India genuinely central to this story, or does
it just mention India in passing? Is this strategically important, or just
noisy? Is there a clean narrative hook a viewer will stop scrolling for?

That's a judgement call, not a keyword match, so this calls Claude
(Sonnet) with a tightly-specified rubric and asks for structured JSON back.
This is the only LLM call in the discovery stage — deliberately isolated
here so it's the one thing you'd inspect/tune if the shortlist ever starts
looking wrong.

Requires ANTHROPIC_API_KEY set in the environment (GitHub Actions secret
in production).
"""

import json
import os
from .models import StoryCandidate, StoryCategory

JUDGE_SYSTEM_PROMPT = """You are the editorial desk for an India-centric \
Tamil-language geopolitics channel. For each story you are given, judge it \
strictly on these axes and return ONLY a JSON object, no other text:

- india_relevance (0-10): Is India genuinely the subject of this story, not \
just mentioned? A story about US-China trade that name-drops India once is \
a 2, not a 6. A story about an India-France defence agreement is a 9-10.
- strategic_importance (0-10): Does this matter to India's security, \
sovereignty, economic strategy, or international standing? Routine \
diplomatic courtesy visits score low; defence deals, border developments, \
major strategic agreements score high.
- virality_signal (0-10): Independent of India-relevance — does this story \
have a clean hook, stakes, or conflict that would make an ordinary viewer \
stop scrolling? Dry procedural news scores low even if strategically real.
- hook_strength (0-10): Can this be opened with one strong sentence that \
creates curiosity, without misrepresenting the facts?
- category: one of defence_deal, diplomacy_bilateral, border_security, \
trade_energy, strategic_tech, multilateral, international_mention, not_relevant
- reasoning: 2-3 sentences explaining the scores, written the way an editor \
would justify a story choice to their team.

Return exactly this JSON shape:
{"india_relevance": <float>, "strategic_importance": <float>, \
"virality_signal": <float>, "hook_strength": <float>, "category": <string>, \
"reasoning": <string>}"""


def judge_candidate(candidate: StoryCandidate) -> StoryCandidate:
    """
    Calls Claude to fill in the judgement fields on a candidate.
    Corroboration is set separately by the discovery layer (it's a simple
    count of independent sources carrying the story, not a judgement call).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. In production this is a GitHub "
            "Actions secret, injected as an environment variable at run time."
        )

    import anthropic  # pip install anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = (
        f"HEADLINE: {candidate.headline}\n"
        f"SOURCE: {candidate.source_name}\n"
        f"SUMMARY: {candidate.summary}"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(raw)

    candidate.india_relevance = float(result["india_relevance"])
    candidate.strategic_importance = float(result["strategic_importance"])
    candidate.virality_signal = float(result["virality_signal"])
    candidate.hook_strength = float(result["hook_strength"])
    candidate.reasoning = result["reasoning"]
    try:
        candidate.category = StoryCategory(result["category"])
    except ValueError:
        candidate.category = StoryCategory.NOT_RELEVANT

    return candidate


def judge_candidates(candidates: list[StoryCandidate]) -> list[StoryCandidate]:
    return [judge_candidate(c) for c in candidates]
