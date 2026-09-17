"""One-off check: ask Groq directly which models this account can actually access right now,
instead of guessing model ID strings from docs/blog posts."""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))

models = client.models.list()
print("Models available to this account:")
for m in sorted(models.data, key=lambda x: x.id):
    print(" -", m.id)
