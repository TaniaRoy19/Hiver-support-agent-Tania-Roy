"""
End-to-end pipeline: runs both baselines and the full LLM-based agent against the real
146-row golden set, then scores everything with the LLM judge.

RESUMABLE at the granular (per-stage) level: if outputs/full_results.csv already exists,
each row's classify/agent/judge stages are checked independently -- a row that was
classified and drafted but not yet judged only re-attempts the judge step, not the whole
row. This matters because free-tier API budgets are limited per day, so a rate-limit
interruption should never throw away partial work.

Methodology note (see decision_log.md): SimpleClassifier is evaluated with 5-fold
cross-validation on the golden set, not trained and tested on the same rows.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from classify import TrivialClassifier, SimpleClassifier, LLMClassifier
from retrieve import ResolutionRetriever
from agent import SupportAgent
from eval.judge import LLMJudge

GOLDEN_PATH = "data/golden_set.csv"
POOL_PATH = "data/retrieval_pool.csv"
OUT_DIR = "outputs"
RESULTS_PATH = f"{OUT_DIR}/full_results.csv"
BRAND = "XboxSupport"

RESULT_FIELDS = [
    "message_id", "text", "difficulty_bucket", "true_intent", "true_escalate",
    "llm_pred_intent", "agent_escalate", "agent_draft_reply", "agent_escalation_reason",
    "agent_grounding_summary", "judge_accuracy", "judge_relevance", "judge_tone",
    "judge_escalation_judgment", "judge_overall", "judge_notes",
]


def load_golden():
    return pd.read_csv(GOLDEN_PATH)


def load_retriever():
    pairs_df = pd.read_csv(POOL_PATH)
    pairs = list(zip(pairs_df["cust_text"], pairs_df["brand_reply"]))
    return ResolutionRetriever(pairs).fit()


def eval_trivial(golden: pd.DataFrame) -> float:
    clf = TrivialClassifier().fit(golden["true_intent"].tolist())
    preds = [clf.predict(t).intent for t in golden["text"]]
    return (pd.Series(preds) == golden["true_intent"].values).mean()


def eval_simple_cv(golden: pd.DataFrame, n_splits: int = 5) -> float:
    texts = golden["text"].tolist()
    labels = golden["true_intent"].tolist()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    correct, total = 0, 0
    for train_idx, test_idx in skf.split(texts, labels):
        clf = SimpleClassifier()
        clf.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
        for i in test_idx:
            pred = clf.predict(texts[i]).intent
            correct += int(pred == labels[i])
            total += 1
    return correct / total


def load_existing_results() -> dict:
    """Returns {message_id: row_dict} for ANY prior partial or full progress on that row."""
    if not os.path.exists(RESULTS_PATH):
        return {}
    try:
        prior = pd.read_csv(RESULTS_PATH)
    except Exception:
        return {}
    return {row["message_id"]: row.to_dict() for _, row in prior.iterrows()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    golden = load_golden()
    print(f"Loaded golden set: {len(golden)} rows")

    prior_results = load_existing_results()
    if prior_results:
        n_fully_done = sum(
            1 for r in prior_results.values()
            if pd.notna(r.get("agent_draft_reply")) and pd.notna(r.get("judge_overall"))
        )
        print(f"Found prior progress: {len(prior_results)} rows touched, {n_fully_done} fully complete.")

    print("\n--- Baseline 1: Trivial (majority-label) classifier ---")
    trivial_acc = eval_trivial(golden)
    print(f"Trivial accuracy: {trivial_acc:.1%}")

    print("\n--- Baseline 2: Simple (TF-IDF + LogReg) classifier, 5-fold CV ---")
    simple_acc = eval_simple_cv(golden)
    print(f"Simple (CV) accuracy: {simple_acc:.1%}")

    print("\n--- Loading retrieval pool ---")
    retriever = load_retriever()

    classifier = LLMClassifier()
    agent = SupportAgent(retriever, brand=BRAND)
    judge = LLMJudge()

    rows_out = []
    n_llm_correct = 0
    n_esc_correct = 0
    n_processed = 0

    print(f"\n--- Processing {len(golden)} rows (per-stage resume) ---")
    for i, row in golden.iterrows():
        mid = row["message_id"]
        prior = prior_results.get(mid, {})

        result_row = {
            "message_id": mid, "text": row["text"], "difficulty_bucket": row["difficulty_bucket"],
            "true_intent": row["true_intent"], "true_escalate": row["true_escalate"],
            "llm_pred_intent": prior.get("llm_pred_intent") if pd.notna(prior.get("llm_pred_intent")) else None,
            "agent_escalate": prior.get("agent_escalate") if pd.notna(prior.get("agent_escalate")) else None,
            "agent_draft_reply": prior.get("agent_draft_reply") if pd.notna(prior.get("agent_draft_reply")) else None,
            "agent_escalation_reason": prior.get("agent_escalation_reason") if pd.notna(prior.get("agent_escalation_reason")) else None,
            "agent_grounding_summary": prior.get("agent_grounding_summary") if pd.notna(prior.get("agent_grounding_summary")) else None,
            "judge_accuracy": prior.get("judge_accuracy") if pd.notna(prior.get("judge_accuracy")) else None,
            "judge_relevance": prior.get("judge_relevance") if pd.notna(prior.get("judge_relevance")) else None,
            "judge_tone": prior.get("judge_tone") if pd.notna(prior.get("judge_tone")) else None,
            "judge_escalation_judgment": prior.get("judge_escalation_judgment") if pd.notna(prior.get("judge_escalation_judgment")) else None,
            "judge_overall": prior.get("judge_overall") if pd.notna(prior.get("judge_overall")) else None,
            "judge_notes": prior.get("judge_notes") if pd.notna(prior.get("judge_notes")) else None,
        }

        # Stage 1: classify (skip if already done)
        if result_row["llm_pred_intent"] is None:
            try:
                cls_result = classifier.predict(row["text"], brand=BRAND)
                result_row["llm_pred_intent"] = cls_result.intent
                time.sleep(0.3)
            except Exception as e:
                print(f"  [warn] classify failed for row {i} (id {mid}): {e}")
        if result_row["llm_pred_intent"] == row["true_intent"]:
            n_llm_correct += 1

        # Stage 2: agent draft + escalate (skip if already done)
        if result_row["agent_draft_reply"] is None:
            try:
                agent_out = agent.handle(row["text"])
                result_row["agent_escalate"] = agent_out.escalate
                result_row["agent_draft_reply"] = agent_out.draft_reply
                result_row["agent_escalation_reason"] = agent_out.escalation_reason
                result_row["agent_grounding_summary"] = "; ".join(
                    g["reply"][:80] for g in agent_out.grounding_used[:2]
                )
                time.sleep(0.3)
            except Exception as e:
                print(f"  [warn] agent failed for row {i} (id {mid}): {e}")
        if result_row["agent_escalate"] == bool(row["true_escalate"]):
            n_esc_correct += 1

        # Stage 3: judge (skip if already done, only run if we have a draft to judge)
        if result_row["judge_overall"] is None and result_row["agent_draft_reply"] is not None:
            try:
                jscore = judge.score(
                    message=row["text"],
                    grounding=result_row["agent_grounding_summary"] or "",
                    draft_reply=result_row["agent_draft_reply"],
                    escalate=result_row["agent_escalate"],
                    escalation_reason=result_row["agent_escalation_reason"],
                )
                result_row["judge_accuracy"] = jscore.accuracy
                result_row["judge_relevance"] = jscore.relevance
                result_row["judge_tone"] = jscore.tone
                result_row["judge_escalation_judgment"] = jscore.escalation_judgment
                result_row["judge_overall"] = jscore.overall
                result_row["judge_notes"] = jscore.notes
                time.sleep(0.3)
            except Exception as e:
                print(f"  [warn] judge failed for row {i} (id {mid}): {e}")

        rows_out.append(result_row)
        n_processed += 1

        pd.DataFrame(rows_out, columns=RESULT_FIELDS).to_csv(RESULTS_PATH, index=False)

        if n_processed % 10 == 0:
            print(f"  ...{n_processed}/{len(golden)} processed")

    out_df = pd.DataFrame(rows_out, columns=RESULT_FIELDS)
    out_df.to_csv(RESULTS_PATH, index=False)

    n_agent_completed = out_df["agent_draft_reply"].notna().sum()
    n_judged = out_df["judge_overall"].notna().sum()
    llm_acc = n_llm_correct / len(golden)
    esc_acc = n_esc_correct / len(golden)

    summary = {
        "trivial_accuracy": trivial_acc,
        "simple_cv_accuracy": simple_acc,
        "llm_classifier_accuracy": llm_acc,
        "escalation_accuracy": esc_acc,
        "n_golden_set": len(golden),
        "n_agent_completed": int(n_agent_completed),
        "n_judged": int(n_judged),
    }
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nFull results saved to {RESULTS_PATH}")
    print(f"Summary saved to {OUT_DIR}/summary.json")
    incomplete = len(golden) - min(n_agent_completed, n_judged)
    if incomplete > 0:
        print(f"\n{incomplete} rows still incomplete -- rerun this script to resume (partial progress is kept per-stage now).")


if __name__ == "__main__":
    main()
