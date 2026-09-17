"""
Build the retrieval pool: real (customer_message, brand_reply) pairs from XboxSupport's
actual resolved threads, saved to data/retrieval_pool.csv.

Important: excludes any customer message whose tweet_id is in one of the golden-set
batch files. Otherwise a golden-set message could be retrieved as its own "past precedent"
when the agent answers it during evaluation -- grading the system against an index that
contains the answer. See decision_log.md.
"""
import pandas as pd
import glob

RAW_CSV = "data/twcs.csv"
BRAND = "XboxSupport"
OUT_PATH = "data/retrieval_pool.csv"


def load_golden_set_ids() -> set:
    """Collect message_ids from any golden_set*.csv files present, labeled or not --
    these must never leak into the retrieval pool."""
    ids = set()
    for path in glob.glob("data/golden_set*.csv"):
        try:
            df = pd.read_csv(path, usecols=["message_id"])
            ids.update(df["message_id"].tolist())
        except Exception as e:
            print(f"Warning: couldn't read {path} ({e}), skipping")
    return ids


def main():
    print(f"Loading {RAW_CSV}...")
    df = pd.read_csv(
        RAW_CSV, low_memory=False,
        usecols=["author_id", "inbound", "text", "in_response_to_tweet_id", "tweet_id"],
    )

    brand_replies = df[(df["author_id"] == BRAND) & (df["inbound"] == False)][
        ["tweet_id", "text", "in_response_to_tweet_id"]
    ].dropna(subset=["in_response_to_tweet_id"])
    brand_replies["in_response_to_tweet_id"] = brand_replies["in_response_to_tweet_id"].astype(int)

    customer_msgs = df[["tweet_id", "text"]].rename(
        columns={"tweet_id": "cust_tweet_id", "text": "cust_text"}
    )

    pairs = brand_replies.merge(
        customer_msgs, left_on="in_response_to_tweet_id", right_on="cust_tweet_id", how="inner"
    )[["cust_tweet_id", "cust_text", "text"]].rename(
        columns={"text": "brand_reply"}
    )

    print(f"Raw pairs found: {len(pairs)}")

    golden_ids = load_golden_set_ids()
    print(f"Excluding {len(golden_ids)} golden-set message_ids from the retrieval pool")
    pairs = pairs[~pairs["cust_tweet_id"].isin(golden_ids)]

    print(f"Final retrieval pool size: {len(pairs)}")
    pairs.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
