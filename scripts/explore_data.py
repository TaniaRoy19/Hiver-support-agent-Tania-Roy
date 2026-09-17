"""
Day 1 exploration: pick a brand, derive intents empirically, check data quality.
Run each section top to bottom — read the comment before each block before running it.
"""
import pandas as pd

# --- Step 1: Load and get oriented ---
# low_memory=False because this dataset has mixed types in some columns (tweet_id
# columns are sometimes read as float vs int inconsistently across rows) -- if we
# don't set this, pandas silently guesses per-chunk and you get dtype mismatches later.
df = pd.read_csv("data/twcs.csv", low_memory=False)

print(df.shape)
print(df.columns.tolist())
print(df.head(3))
