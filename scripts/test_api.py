"""Quick sanity check: confirm the Groq API key works before building anything bigger on top."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classify import LLMClassifier

clf = LLMClassifier()
result = clf.predict("My xbox won't turn on at all, tried everything", brand="XboxSupport")
print("intent:", result.intent)
print("confidence:", result.confidence)
print("method:", result.method)
print("\nIf you see a sensible intent above (e.g. hardware_issue), the API key and setup are working.")
