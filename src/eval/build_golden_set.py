"""
Day 1, final step: build the golden set sampling pool.

Composite difficulty score per message (calibrated against real XboxSupport data —
see decision log for the percentile checks behind each threshold):
  +3  matches 0 of the 7 intent keyword groups
  +2  matches 2+ intent keyword groups
  +1  emotional marker: caps_ratio > 0.3, OR >=1 exclamation mark, OR a frustration word
  +1  unusual length: <=5 words or >=45 words

Score -> bucket: 0 = typical, 1 = edge, 2 = ambiguous, 3+ = adversarial.
Target sample sizes: 80 typical / 50 ambiguous / 40 edge / 30 adversarial = 200 total.

This script only SAMPLES and writes an empty-labels CSV. You do the actual labeling by hand
in a spreadsheet tool -- that's the required "hand-labeled" part of the deliverable.
"""
import pandas as pd

RAW_CSV = "data/twcs.csv"
BRAND = "XboxSupport"
OUT_PATH = "data/golden_set.csv"
SEED = 42

TARGET_COUNTS = {
    "typical": 80,
    "ambiguous": 50,
    "edge": 40,
    "adversarial": 30,
}

INTENT_KEYWORDS = {
    "hardware_issue": ["turn on", "wont turn", "won't turn", "power", "reset", "overheat", "disc"],
    "game_crash_bug": ["crash", "won't load", "wont load", "freeze", "froze", "bug", "glitch", "broken"],
    "account_data_issue": ["gamerscore", "account", "data", "save file", "deleted", "lost my"],
    "refund_request": ["refund", "money back", "charged"],
    "network_connectivity": ["server", "network", "connect", "online", "down", "offline"],
    "how_to_question": ["how do i", "how to", "parental control", "can i", "is there a way"],
    "code_activation": ["code", "redeem", "activate", "key"],
}

FRUSTRATION_WORDS = ["fuck", "shit", "ass", "hate", "worst", "ridiculous", "unacceptable", "pathetic"]


def caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0


def matched_intent_groups(text_lower: str) -> int:
    return sum(1 for kws in INTENT_KEYWORDS.values() if any(k in text_lower for k in kws))


def is_emotional(text: str, text_lower: str) -> bool:
    return (
        caps_ratio(text) > 0.3
        or text.count("!") >= 1
        or any(w in text_lower for w in FRUSTRATION_WORDS)
    )


def is_unusual_length(text: str) -> bool:
    n = len(text.split())
    return n <= 5 or n >= 45


def difficulty_score(text: str) -> int:
    text_lower = text.lower()
    n_groups = matched_intent_groups(text_lower)

    score = 0
    if n_groups == 0:
        score += 3
    elif n_groups >= 2:
        score += 2
    if is_emotional(text, text_lower):
        score += 1
    if is_unusual_length(text):
        score += 1
    return score


def score_to_bucket(score: int) -> str:
    if score == 0:
        return "typical"
    if score == 1:
        return "edge"
    if score == 2:
        return "ambiguous"
    return "adversarial"


def build_pool() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_CSV, low_memory=False,
        usecols=["author_id", "inbound", "text", "in_response_to_tweet_id", "tweet_id"],
    )
    brand_reply_targets = df[(df["author_id"] == BRAND) & (df["inbound"] == False)][
        "in_response_to_tweet_id"
    ].dropna().astype(int)
    customer_msgs = df[df["tweet_id"].isin(brand_reply_targets)][["tweet_id", "text"]].copy()

    customer_msgs["difficulty_score"] = customer_msgs["text"].apply(difficulty_score)
    customer_msgs["difficulty_bucket"] = customer_msgs["difficulty_score"].apply(score_to_bucket)
    return customer_msgs


def sample_golden_set(pool: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    parts = []
    for bucket, target_n in TARGET_COUNTS.items():
        bucket_pool = pool[pool["difficulty_bucket"] == bucket]
        n = min(target_n, len(bucket_pool))
        if n < target_n:
            print(f"WARNING: only {len(bucket_pool)} available for bucket '{bucket}', "
                  f"wanted {target_n}")
        parts.append(bucket_pool.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def main():
    print(f"Loading and scoring messages for {BRAND}...")
    pool = build_pool()
    print(f"Pool size: {len(pool)}")
    print(pool["difficulty_bucket"].value_counts())

    golden = sample_golden_set(pool)
    golden = golden.rename(columns={"tweet_id": "message_id"})
    golden["true_intent"] = ""
    golden["gold_reply_notes"] = ""
    golden["true_escalate"] = ""
    golden["true_escalate_reason"] = ""

    cols = ["message_id", "text", "difficulty_bucket", "difficulty_score",
            "true_intent", "gold_reply_notes", "true_escalate", "true_escalate_reason"]
    golden = golden[cols].sample(frac=1, random_state=SEED + 1).reset_index(drop=True)
    golden.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(golden)} rows to {OUT_PATH} -- ready for hand labeling.")
    print(golden["difficulty_bucket"].value_counts())


if __name__ == "__main__":
    main()
