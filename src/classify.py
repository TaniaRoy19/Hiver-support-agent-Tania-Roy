"""
Intent classification for customer support messages.

Two implementations on purpose:
- SimpleClassifier: TF-IDF + logistic regression. This is baseline #2 (required by the brief),
  and also doubles as a fast sanity check before burning LLM calls.
- LLMClassifier: prompts an LLM (via Groq's free API) with the intent taxonomy.

Keep both -- the report needs a head-to-head comparison, not just the LLM result.
"""
import json
import os
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

from groq_utils import call_json_with_retry

load_dotenv()  # reads GROQ_API_KEY from a .env file in the project root

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# openai/gpt-oss-120b -- Groq's current recommended model as of Sep 2026.
# llama-3.3-70b-versatile was deprecated by Groq on 2026-08-16 -- do not use it.
DEFAULT_MODEL = "qwen/qwen3.8-27b"  # verified against the real account model list via scripts/list_models.py -- the earlier qwen3.6-27b guess was wrong/inaccessible.

# Derived from XboxSupport customer messages (see decision log + README "Intents" section
# for the frequency check that validated this taxonomy against real data).
DEFAULT_INTENTS = [
    "hardware_issue",       # console power/physical issues, resets
    "game_crash_bug",       # games/apps won't load, crash, broken after update
    "account_data_issue",   # lost data, gamerscore anomalies, account access
    "refund_request",
    "network_connectivity", # server status, connection drops
    "how_to_question",      # feature/settings questions, parental controls
    "code_activation",      # redeem codes, activation problems
    "other_unclear",        # noise, off-topic, ambiguous, unresolvable from text alone
]


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    method: str


class TrivialClassifier:
    """Trivial baseline #1: always predicts the most frequent intent in the training set.
    Required by the brief as the floor the agent has to beat."""

    def __init__(self):
        self.majority_label = None

    def fit(self, labels):
        from collections import Counter
        self.majority_label = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, text: str) -> ClassificationResult:
        if self.majority_label is None:
            raise RuntimeError("TrivialClassifier must be fit() before predict()")
        return ClassificationResult(intent=self.majority_label, confidence=1.0, method="trivial_majority")


class SimpleClassifier:
    """TF-IDF + LogisticRegression baseline. No LLM calls."""

    def __init__(self, intents=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.intents = intents or DEFAULT_INTENTS
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self._fitted = True
        return self

    def predict(self, text: str) -> ClassificationResult:
        if not self._fitted:
            raise RuntimeError("SimpleClassifier must be fit() before predict()")
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X).max()
        return ClassificationResult(intent=pred, confidence=float(proba), method="tfidf_logreg")


class LLMClassifier:
    def __init__(self, intents=None, model=DEFAULT_MODEL):
        self.intents = intents or DEFAULT_INTENTS
        self.model = model
        # OpenAI SDK pointed at Groq's OpenAI-compatible endpoint -- reads GROQ_API_KEY
        self.client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ.get("GROQ_API_KEY"))

    def predict(self, text: str, brand: str = "") -> ClassificationResult:
        prompt = f"""You are classifying a customer support message from a customer to {brand or 'a company'}.

Intent categories: {json.dumps(self.intents)}

Message: "{text}"

Respond with ONLY a JSON object: {{"intent": "<one of the categories>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""

        # gpt-oss models are reasoning models and need reasoning hidden or they burn their
        # token budget on hidden chain-of-thought before writing the JSON answer. Other
        # model families (e.g. qwen) don't support/need this param.
        extra = {"reasoning_format": "hidden"} if "gpt-oss" in self.model else {}
        parsed = call_json_with_retry(self.client, self.model, prompt, max_tokens=500, extra_body=extra)
        return ClassificationResult(
            intent=parsed["intent"], confidence=parsed.get("confidence", 0.5), method="llm"
        )
