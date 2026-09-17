# Hiver SDE Intern — AI Support Agent

## What this is
An AI support agent for **XboxSupport** that classifies customer message intent, drafts a
reply grounded in real past resolutions, and decides auto-handle vs. escalate — evaluated
against a 146-row hand-labeled golden set with LLM-as-judge scoring, a judge-human agreement
check, and honest failure analysis.

**See `report.md` for the full write-up** (problem framing, results, failure analysis, the
"what's misleading about my headline number" section, and what I'd do with one more week).
This README covers setup and reproduction only.

## Status
✅ Complete. Full pipeline run on all 146 golden-set rows. Headline results: LLM classifier
74.7% intent accuracy (vs. 17.8% trivial / 37.7% simple baselines); escalation accuracy 52.0%
(a real finding, not a bug — see report.md Section 4).

## Repro (target: <15 min)

**`outputs/full_results.csv` and `outputs/summary.json` are committed** with the completed
run (all 146 rows). This is intentional, not an oversight: the pipeline makes ~450 real API
calls and is genuinely resumable — it skips any row already present in `outputs/full_results.csv`
— so running it against the committed results reproduces the exact headline numbers in under a
minute (measured: 33 seconds) instead of requiring a full from-scratch re-run every time.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Groq API key at https://console.groq.com (no cost, no card for basic use)
#    Create a .env file in the project root containing:
#    GROQ_API_KEY=your-key-here

# 3. Download the raw dataset (not bundled here, ~500MB):
#    Kaggle "Customer Support on Twitter" — thoughtvector/customer-support-on-twitter
#    Place twcs.csv at data/twcs.csv (directly in data/, not in a subfolder)

# 4. Build the retrieval pool (real customer/brand-reply pairs, golden-set rows excluded)
python scripts/build_retrieval_pool.py

# 5. Run the pipeline — with outputs/full_results.csv already present, this verifies and
#    reprints the committed results rather than re-running ~450 API calls (measured: 33s)
python scripts/run_pipeline.py
```

**To verify a genuine from-scratch run instead** (not required, but possible): delete
`outputs/full_results.csv` before step 5. Expect several minutes, not seconds, since every
row's classification, draft, and judge score will be freshly generated via API calls.

### Optional: regenerating the golden set from scratch
The golden set is already built and labeled (`data/golden_set.csv`). To see how it was
sampled: `python src/eval/build_golden_set.py` (writes stratified, *unlabeled* batches —
hand-labeling is the actual deliverable and isn't automated).

### Optional: judge-human agreement check
Already computed (see `report.md` Section 5) using `data/judge_agreement_subsample.csv` — a
35-row stratified sample scored blind against the judge's own scores. The comparison logic is
in `src/eval/judge.py`'s `judge_human_agreement()`.

## Structure
```
data/
  twcs.csv                     # raw dataset (not bundled — see step 3 above)
  golden_set.csv                # 146 hand-labeled rows (the real deliverable)
  golden_set_batch[1-4].csv      # the golden set in labeling-sized chunks (source of golden_set.csv)
  retrieval_pool.csv              # real (customer_msg, brand_reply) pairs, golden-set rows excluded
  judge_agreement_subsample.csv    # 35-row judge-vs-human scoring comparison
src/
  classify.py            # TrivialClassifier, SimpleClassifier (baselines), LLMClassifier
  retrieve.py              # TF-IDF retrieval over real resolved threads
  agent.py                   # classify -> retrieve -> draft -> escalate (2 separate LLM calls)
  groq_utils.py                 # shared robust JSON-calling helper (retry + lenient parsing)
  eval/
    build_golden_set.py      # stratified difficulty-based sampling
    judge.py                   # LLM-as-judge + judge_human_agreement()
    metrics.py                    # bootstrap CIs, bucket breakdowns, baseline comparison
scripts/
  build_retrieval_pool.py   # builds data/retrieval_pool.csv from real thread structure
  run_pipeline.py             # end-to-end runner, resumable per-stage
  list_models.py                 # utility: lists models actually available to your API key
decision_log.md        # 25 numbered decisions, including real incidents and how they were fixed
report.md              # the actual submission report — start here
```

## Brand choice
**XboxSupport.** Chosen over AmazonHelp, AppleSupport, SpotifyCares, and Delta after comparing
volume + reply substantiveness (not just a deflection keyword rate — see decision log #7 for
why that number alone was misleading). Narrow single-ecosystem product surface, 24,557 brand
replies, replies show genuine troubleshooting rather than deflection.

## Intents
Derived from reading real XboxSupport customer messages, then sanity-checked with a rough
keyword-frequency pass (see decision log #8) before locking in:
`hardware_issue`, `game_crash_bug`, `account_data_issue`, `refund_request`,
`network_connectivity`, `how_to_question`, `code_activation`, `other_unclear`.

## Baselines
1. Trivial: most-frequent-intent classifier → `TrivialClassifier` in `src/classify.py`
2. Simple: TF-IDF + logistic regression, 5-fold cross-validated → `SimpleClassifier`
Both required by the spec — see `report.md` Section 3 for how much the LLM-based agent beats
each one by, and Section 4 for why the headline accuracy number alone doesn't tell the whole
story.

## A note on the model provider
Built against Groq's free API (OpenAI-compatible endpoint), not Anthropic's, since a paid key
wasn't available — see decision log #11. Currently uses `qwen/qwen3.8-27b`. If you rerun this
and hit a "model not found" or quota error, run `python scripts/list_models.py` to see exactly
which models your account can currently access, and update `DEFAULT_MODEL` in `src/classify.py`,
`src/agent.py`, and `src/eval/judge.py` accordingly (see decision log #18-20 for the full story
of why this needed troubleshooting more than once).
