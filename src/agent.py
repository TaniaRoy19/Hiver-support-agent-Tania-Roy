"""
The support agent: classify -> retrieve grounding -> draft reply -> escalation decision.

Escalation is a decision WITH a stated reason -- not a side effect. Implemented as TWO
separate LLM calls (not one combined prompt): draft first, then a separate escalation
judgment that sees the finished draft. This avoids the AI grading its own work in the same
breath it wrote it, and lets draft quality and escalation judgment be evaluated and
failure-analyzed independently. See decision_log.md #3.
"""
import json
import os
from dataclasses import dataclass, asdict

from openai import OpenAI
from dotenv import load_dotenv

from groq_utils import call_json_with_retry

load_dotenv()

from classify import LLMClassifier, ClassificationResult
from retrieve import ResolutionRetriever

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# openai/gpt-oss-120b -- Groq's current recommended model as of Sep 2026.
# llama-3.3-70b-versatile was deprecated by Groq on 2026-08-16 -- do not use it.
DEFAULT_MODEL = "qwen/qwen3.8-27b"  # verified against the real account model list via scripts/list_models.py -- the earlier qwen3.6-27b guess was wrong/inaccessible.


@dataclass
class AgentOutput:
    message: str
    intent: str
    intent_confidence: float
    draft_reply: str
    escalate: bool
    escalation_reason: str
    grounding_used: list


class SupportAgent:
    def __init__(self, retriever: ResolutionRetriever, brand: str, model=DEFAULT_MODEL):
        self.retriever = retriever
        self.brand = brand
        self.classifier = LLMClassifier()
        self.client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def _call_json(self, prompt: str, max_tokens: int = 600) -> dict:
        extra = {"reasoning_format": "hidden"} if "gpt-oss" in self.model else {}
        return call_json_with_retry(self.client, self.model, prompt, max_tokens=max_tokens, extra_body=extra)

    def _draft_reply(self, message: str, intent: str, grounding_text: str) -> str:
        prompt = f"""You are drafting a support reply for {self.brand}.

Customer message: "{message}"
Classified intent: {intent}

Here is how {self.brand} has handled similar issues before:
{grounding_text}

Draft a reply consistent with how this brand has actually resolved similar issues.

Respond with ONLY JSON:
{{"draft_reply": "..."}}"""
        return self._call_json(prompt, max_tokens=500)["draft_reply"]

    def _decide_escalation(self, message: str, intent: str, grounding_text: str, draft_reply: str) -> tuple[bool, str]:
        prompt = f"""You are reviewing a drafted support reply for {self.brand} to decide whether it can be
auto-sent to the customer or must be escalated to a human agent.

Customer message: "{message}"
Classified intent: {intent}

Here is how {self.brand} has handled similar issues before:
{grounding_text}

Drafted reply:
"{draft_reply}"

Escalate if the grounding examples are weak/contradictory, if the issue involves money/account
security, or if you're not confident the draft is accurate.

Respond with ONLY JSON:
{{"escalate": true/false, "escalation_reason": "..."}}"""
        result = self._call_json(prompt, max_tokens=400)
        return result["escalate"], result["escalation_reason"]

    def handle(self, message: str) -> AgentOutput:
        cls: ClassificationResult = self.classifier.predict(message, brand=self.brand)
        grounding = self.retriever.retrieve(message, k=3)

        grounding_text = "\n".join(
            f"- Similar past message: \"{m}\" -> Brand's actual reply: \"{r}\" (similarity: {s:.2f})"
            for m, r, s in grounding
        )

        draft_reply = self._draft_reply(message, cls.intent, grounding_text)
        escalate, escalation_reason = self._decide_escalation(message, cls.intent, grounding_text, draft_reply)

        return AgentOutput(
            message=message,
            intent=cls.intent,
            intent_confidence=cls.confidence,
            draft_reply=draft_reply,
            escalate=escalate,
            escalation_reason=escalation_reason,
            grounding_used=[{"msg": m, "reply": r, "score": s} for m, r, s in grounding],
        )

    def handle_batch(self, messages: list[str]) -> list[dict]:
        return [asdict(self.handle(m)) for m in messages]
