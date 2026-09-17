"""
Interactive terminal labeler -- same manual judgment as the spreadsheet, less friction.
Run it, answer prompts with keystrokes, it saves after every row so you can quit anytime
and resume later without losing progress.

Usage:
    python scripts/label_golden_set.py
"""
import pandas as pd
import sys

PATH = "data/golden_set.csv"

INTENTS = [
    "hardware_issue", "game_crash_bug", "account_data_issue", "refund_request",
    "network_connectivity", "how_to_question", "code_activation", "other_unclear",
]


def prompt_intent() -> str:
    print("\nIntent:")
    for i, name in enumerate(INTENTS, 1):
        print(f"  {i}. {name}")
    while True:
        choice = input("Pick 1-8 (or 's' to skip this row): ").strip().lower()
        if choice == "s":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            return INTENTS[int(choice) - 1]
        print("Invalid input, try again.")


def prompt_escalate() -> bool:
    while True:
        choice = input("Escalate to human? (y/n): ").strip().lower()
        if choice in ("y", "n"):
            return choice == "y"
        print("Invalid input, try again.")


def main():
    df = pd.read_csv(PATH)

    remaining = df[df["true_intent"].isna() | (df["true_intent"] == "")]
    print(f"{len(remaining)} of {len(df)} rows still need labeling.\n")

    if len(remaining) == 0:
        print("All rows already labeled.")
        return

    for idx in remaining.index:
        row = df.loc[idx]
        print("=" * 70)
        print(f"[{row['message_id']}] bucket={row['difficulty_bucket']} score={row['difficulty_score']}")
        print(f'"{row["text"]}"')

        intent = prompt_intent()
        if intent is None:
            print("Skipped.")
            continue

        escalate = prompt_escalate()
        reason = input("Escalation reason (short, or blank): ").strip()
        notes = input("Reply notes -- what a good reply must/must not say (or blank): ").strip()

        df.loc[idx, "true_intent"] = intent
        df.loc[idx, "true_escalate"] = escalate
        df.loc[idx, "true_escalate_reason"] = reason
        df.loc[idx, "gold_reply_notes"] = notes

        df.to_csv(PATH, index=False)  # save after every row -- safe to Ctrl+C anytime
        print("Saved.")

    print("\nDone with all remaining rows.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped early -- progress saved. Run again to resume where you left off.")
        sys.exit(0)
