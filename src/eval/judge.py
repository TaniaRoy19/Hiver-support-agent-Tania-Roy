"""
LLM-as-judge for reply quality, WITH a check on whether the judge is trustworthy.

The brief explicitly asks for "evidence of how well your judge agrees with a human."
So: you (Tania) hand-score a subsample (suggest 30-40 of the golden set) on the same
rubric BEFORE or independently of running the judge, then compute agreement
(e.g. Cohen's kappa, or simple % exact + % within-1-point). Report both numbers.
If agreement is weak, say so in the report — that's not a failure, that's rigor.
"""
import json
import os
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

from groq_utils import call_json_with_retry

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# openai/gpt-oss-120b -- Groq's current recommended model as of Sep 2026.
# llama-3.3-70b-versatile was deprecated by Groq on 2026-08-16 -- do not use it.
DEFAULT_MODEL = "qwen/qwen3.8-27b"  # verified against the real account model list via scripts/list_models.py -- the earlier qwen3.6-27b guess was wrong/inaccessible.

JUDGE_RUBRIC = """Score the AI-drafted support reply on a 1-5 scale for each dimension:

1. Accuracy: does it correctly reflect what the brand would actually say/do (based on the
   grounding examples), with no invented policy or false promises?
2. Relevance: does it address the actual customer message (not a generic templated reply)?
3. Tone: is it appropriate for a support context (not curt, not falsely cheerful given
   the customer's frustration)?
4. Escalation judgment: was the auto-handle/escalate decision reasonable given the message
   and grounding?

Customer message: {message}
Grounding used: {grounding}
Draft reply: {draft_reply}
Escalate decision: {escalate} — reason: {escalation_reason}

Respond with ONLY JSON:
{{"accuracy": 1-5, "relevance": 1-5, "tone": 1-5, "escalation_judgment": 1-5,
  "overall": 1-5, "notes": "one sentence on the biggest issue, if any"}}"""


@dataclass
class JudgeScore:
    accuracy: int
    relevance: int
    tone: int
    escalation_judgment: int
    overall: int
    notes: str


class LLMJudge:
    def __init__(self, model=DEFAULT_MODEL):
        self.client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def score(self, message, grounding, draft_reply, escalate, escalation_reason) -> JudgeScore:
        prompt = JUDGE_RUBRIC.format(
            message=message, grounding=grounding, draft_reply=draft_reply,
            escalate=escalate, escalation_reason=escalation_reason,
        )
        extra = {"reasoning_format": "hidden"} if "gpt-oss" in self.model else {}
        parsed = call_json_with_retry(self.client, self.model, prompt, max_tokens=500, extra_body=extra)

        # Models occasionally misspell a field name (seen: "escalation_justment" instead of
        # "escalation_judgment"). Recover the real value via fuzzy key matching rather than
        # discarding it -- fabricating a fake neutral score would quietly corrupt real judge
        # data, which matters for the judge-human agreement check.
        import difflib
        known_fields = ["accuracy", "relevance", "tone", "escalation_judgment", "overall", "notes"]
        clean = {}
        used_keys = set()
        for field in known_fields:
            if field in parsed:
                clean[field] = parsed[field]
                used_keys.add(field)
            else:
                remaining = [k for k in parsed if k not in used_keys]
                match = difflib.get_close_matches(field, remaining, n=1, cutoff=0.6)
                if match:
                    clean[field] = parsed[match[0]]
                    used_keys.add(match[0])
        missing = [f for f in known_fields if f not in clean]
        for field in missing:
            if field == "notes":
                clean[field] = ""
            else:
                clean[field] = 3  # only reached if truly no usable value found anywhere
        if missing:
            clean["notes"] = (clean.get("notes", "") + f" [warning: fields {missing} defaulted, not real model output]").strip()
        return JudgeScore(**clean)


def judge_human_agreement(judge_scores: list[int], human_scores: list[int]) -> dict:
    """
    Compute simple agreement metrics between judge and human scores on the same items.
    Use this on your ~30-40 hand-scored subsample.
    """
    assert len(judge_scores) == len(human_scores)
    n = len(judge_scores)
    exact = sum(j == h for j, h in zip(judge_scores, human_scores)) / n
    within_1 = sum(abs(j - h) <= 1 for j, h in zip(judge_scores, human_scores)) / n

    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(judge_scores, human_scores)
    except Exception:
        kappa = None

    return {"n": n, "exact_agreement": exact, "within_1_agreement": within_1, "cohen_kappa": kappa}
