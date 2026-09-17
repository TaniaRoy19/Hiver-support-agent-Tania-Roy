"""
Retrieval of similar, historically-resolved tickets to ground the drafted reply.

The Twitter support dataset has response_tweet_id / in_response_to_tweet_id links,
so "resolved" threads can be reconstructed: customer message -> brand's actual reply.
That reply pair IS the grounding source — this is not a generic RAG-over-docs setup,
it's retrieval over real resolution precedent, which is closer to what Hiver is asking for.

TODO once dataset is loaded:
1. Build (customer_msg, brand_reply) pairs per brand from the thread structure.
2. Filter to pairs where the brand reply looks like an actual resolution, not just
   "we're looking into it" (this filtering choice belongs in the decision log).
3. Embed customer messages, index them, retrieve top-k nearest to a new incoming message.
"""
import numpy as np


class ResolutionRetriever:
    def __init__(self, pairs: list[tuple[str, str]]):
        """pairs: list of (past_customer_message, brand_reply)"""
        self.pairs = pairs
        self._vectorizer = None
        self._matrix = None

    def fit(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([p[0] for p in self.pairs])
        return self

    def retrieve(self, message: str, k: int = 3) -> list[tuple[str, str, float]]:
        """Returns top-k (past_message, brand_reply, similarity_score)."""
        if self._vectorizer is None:
            raise RuntimeError("Call fit() first")
        from sklearn.metrics.pairwise import cosine_similarity

        q = self._vectorizer.transform([message])
        sims = cosine_similarity(q, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:k]
        return [(self.pairs[i][0], self.pairs[i][1], float(sims[i])) for i in top_idx]

    # NOTE: TF-IDF retrieval is the honest starting point given the timeline.
    # If time allows, swap in embedding-based retrieval (e.g. sentence-transformers)
    # and report whether it actually changes downstream reply quality — don't upgrade
    # this just because embeddings sound better; the report should show the delta.
