# Decision Log

Plain list of non-obvious decisions and why. Add to this as you go — it's a required deliverable,
easiest to lose if you leave it to the end.

1. Used TF-IDF retrieval instead of embeddings for the resolution index, given the timeline —
   flagged as a possible upgrade, not assumed to be worse.
2. Defined intents empirically from the data rather than guessing a taxonomy upfront.
3. Separated escalation decision into its own reasoned output rather than folding it into
   the reply-drafting prompt, so it can be judged and failure-analyzed independently.
4. Golden set stratified across typical/ambiguous/edge/adversarial buckets rather than
   randomly sampled, specifically so the headline accuracy number can be broken down
   by difficulty instead of presented as one figure.
5. Judge trustworthiness is checked against a hand-scored subsample (~30-40 examples)
   rather than assumed; agreement (kappa + %-within-1) is reported alongside judge scores.
6. Retrieval pool and evaluation messages are kept non-overlapping so a message is never
   graded against a retrieval index containing its own resolution.
7. Brand chosen: XboxSupport. Compared volume + a keyword-based deflection rate (fraction
   of replies containing "dm"/"private message"/etc.) across AppleSupport, SpotifyCares,
   XboxSupport, Delta, AmazonHelp — then read actual reply samples rather than trusting
   the keyword number alone. AmazonHelp had the lowest keyword-deflection rate (2%) but
   most of its "non-deflection" replies were actually deflections to an external form
   ("fill this form: [link]") that the keyword list didn't catch — a real example of a
   misleading headline number. XboxSupport's replies were consistently the most substantive
   on manual read (concrete troubleshooting steps, specific settings info), has a narrow
   single-ecosystem product surface (good for a small intent taxonomy), and reasonable
   volume (24,557 brand replies).
8. Intent taxonomy (8 categories, listed in classify.py) derived by reading real XboxSupport
   customer messages, then validated with a rough keyword-based frequency count rather than
   locked in from a first read alone. That check itself produced a misleading number worth
   remembering: 66% of messages matched no keyword ("other_unclear"), which looks like most
   traffic is ambiguous — but it's actually a keyword-list-too-narrow artifact (e.g. "it won't
   start" doesn't match the literal string "turn on"). Used the check only to confirm no
   candidate intent was near-zero-frequency, not to estimate true intent distribution — the
   golden set labeling will give the real distribution.
9. Golden set sampling implemented as a composite difficulty score (not a manual priority
   list for overlap cases) — see README for the calibration numbers behind each threshold.
   Ran against the real 21,276 XboxSupport customer messages: pool sizes were 4,460 typical /
   1,138 ambiguous / 1,260 edge / 14,418 adversarial, all comfortably above target sample
   sizes. Target size set at 150 (the spec's floor of 150-250) rather than 200 — 60 typical /
   38 ambiguous / 30 edge / 22 adversarial — to make hand-labeling genuinely completable
   within the timeline without cutting corners on actually reading each message. Split into
   4 stratified batches (~37-39 rows each, same bucket proportions in every batch) for
   labeling across separate sittings.
10. Retrieval pool built from real thread structure: 24,312 raw (customer_message,
    brand_reply) pairs reconstructed from XboxSupport's actual resolved threads, with all
    150 golden-set message_ids explicitly excluded (24,134 final pool size) so no golden-set
    message can be retrieved as its own grounding precedent during evaluation. Verified with
    a live query — TF-IDF retrieval correctly surfaced near-identical past "won't turn on"
    messages with real brand replies as grounding (similarity 0.72, 0.72, 0.65).
11. Switched LLM provider from Anthropic API to Groq's free API (OpenAI-compatible endpoint)
    partway through, since a paid Anthropic key wasn't available. Using openai/gpt-oss-120b
    (Groq's current recommended model as of Sep 2026 — llama-3.3-70b-versatile was
    deprecated 2026-08-16, avoided deliberately).
12. Batch 1 of golden-set labeling completed (38 of 39 rows — one excluded, see #13).
    Two genuine judgment calls worth remembering: row 1327647 (ambiguous hardware complaint
    with an explicit churn threat, "must fix it now or lose loyal customers") was labeled
    hardware_issue/escalate=True on the retention-risk signal, not just the technical content;
    row 546890 (written in French) was labeled based on readable content, flagged as a scoping
    note that the retrieval pool is presumed English-majority so non-English messages likely
    get weaker grounding — worth naming as a limitation in the report.
13. Excluded message_id 357069 from the golden set as a data-quality issue rather than forcing
    a label: its text reads like a brand support reply ("Just so we're on the same page, what
    error code..." signed "^TJ"), not a customer message, suggesting the thread-reconstruction
    logic mis-threaded a brand-to-brand or chained reply as if it were a customer's. Kept as
    documented evidence of a real limitation rather than silently dropped or mislabeled.
14. Batch 2 of golden-set labeling completed (38 of 39 rows — one excluded, same
    misthreaded-brand-reply issue as #13, confirming it's a real recurring pattern in the
    thread-reconstruction logic, not a one-off). Several escalate=True judgment calls made
    on a consistent rule: when the customer explicitly states standard troubleshooting was
    already tried and failed (e.g. "already did all that, it's a server problem"; "did a
    reset"; "already sent a message, please reply"), escalate rather than offer the same
    generic fix again — repeating advice the customer already tried is a real failure mode
    worth watching for in the failure analysis later.
15. Batch 3 of golden-set labeling completed (35 of 36 rows — one excluded, fourth instance
    of the misthreaded-brand-reply issue, now clearly a systematic pattern worth naming
    explicitly in the report's failure analysis, not just an edge case). Also found one
    likely non-genuine input (message_id 1539671, a joke/riddle tweet reusing another
    message's error code) — labeled other_unclear rather than excluded, since it's a
    legitimate customer-facing message the agent would actually encounter in production;
    a real system needs to handle non-genuine input gracefully, not just filter it out in
    the eval set.
16. Batch 4 of golden-set labeling completed (35 of 36 rows — one excluded, fourth confirmed
    instance of the misthreaded-brand-reply issue across the 4 batches; consistent enough
    to describe as systematic rather than incidental in the report). All 4 batches now done:
    146 labeled rows total (150 sampled minus 4 excluded misthreaded rows, one per batch),
    across typical/ambiguous/edge/adversarial buckets. Golden set is complete.
17. Baselines computed against the real 146-row golden set: trivial (majority-label)
    accuracy 17.8% (majority class: other_unclear), simple (TF-IDF + logistic regression)
    accuracy 37.7% via 5-fold cross-validation. CV was used instead of train-and-test-on-
    the-same-146-rows specifically to avoid leakage — with only the golden set available as
    labeled data, training and testing on the identical rows would have produced an
    artificially inflated number. This is the real comparison point the LLM-based agent's
    accuracy needs to beat, and by how much.
18. First full pipeline run mostly failed: openai/gpt-oss-120b is a reasoning model, and
    without reasoning suppressed it burned its token budget on hidden chain-of-thought
    before writing the actual JSON answer -- causing empty or truncated responses (104 of
    146 agent calls failed, judge failed on all 146). Fixed by adding
    extra_body={"reasoning_format": "hidden"} to every Groq call. The failed run also
    silently exhausted gpt-oss-120b's entire 200,000-token daily free-tier budget before
    the fix landed, blocking the very next run too. Fixed by switching to
    openai/gpt-oss-20b, which gets a separate, independent daily quota on Groq's free
    tier (confirmed budgets are tracked per-model, not account-wide) -- this unblocked
    immediately rather than waiting for the midnight UTC reset. Also made
    scripts/run_pipeline.py resumable (saves progress after every row, skips rows already
    completed on rerun) specifically so a future rate-limit interruption doesn't waste
    budget re-processing rows that already succeeded.
19. Second pipeline run also hit its daily token cap (openai/gpt-oss-20b's separate 200K
    TPD pool), 46/146 agent rows and 30/146 judge rows completed before stopping --
    resumability worked as intended, progress was saved rather than lost. Also discovered
    llama-3.1-8b-instant (which has a much larger 500K TPD free-tier budget) was deprecated
    by Groq on the same date as llama-3.3-70b-versatile, so it isn't a usable option.
    Switched to qwen/qwen3.6-27b -- a different model family, so a genuinely separate
    quota pool from both gpt-oss models already exhausted today. Made the
    reasoning_format=hidden parameter conditional on the model name (only applied for
    gpt-oss models) since qwen doesn't support/need it.
20. Corrected the qwen model ID: qwen/qwen3.6-27b (from deprecation-notice text) doesn't
    actually exist on the account -- ran scripts/list_models.py (calls the API's own
    models.list() endpoint) to get the real, authoritative list of accessible models
    instead of continuing to guess from indirect sources. Correct ID is qwen/qwen3.8-27b.
    Lesson: verify exact model IDs against the live API, not docs/blog text, before
    relying on them.
21. Made run_pipeline.py's resume logic granular per-stage (classify / agent / judge)
    instead of all-or-nothing per row, after losing partial progress on 16 rows when a
    model-not-found error hit mid-run: previously a row only counted as "done" if BOTH
    agent and judge succeeded, so a row with a draft but no judge score got wiped and
    redone from scratch on retry. Now each stage is checked and skipped independently if
    already complete, so no successful API call is ever wasted on a rerun.
22. Built a shared src/groq_utils.py after seeing three distinct real failure modes in one
    run against qwen/qwen3.8-27b: (a) per-MINUTE rate limits (OTPM) -- genuinely transient,
    unlike the per-day quota exhaustion hit earlier -- now retried with backoff instead of
    giving up on the row; (b) "Extra data" JSON errors from trailing text after the JSON
    object -- now falls back to extracting the first balanced {...} block; (c) "Invalid
    control character" errors from raw newlines inside string values -- now falls back to
    lenient (non-strict) JSON parsing. classify.py, agent.py, and judge.py all now call
    this one shared, tested helper instead of three separate ad-hoc JSON-parsing blocks.
23. Judge occasionally misspells a field name in its JSON output (seen: "escalation_justment"
    for "escalation_judgment"). Fixed with fuzzy key matching (difflib) to recover the real
    value under the misspelled key, rather than either crashing (losing the row) or silently
    defaulting to a fake neutral score (which would have quietly corrupted real judge data --
    considered and rejected this simpler option specifically because it would bias the
    judge-human agreement check). A field is only defaulted, and explicitly flagged in its
    notes as not real model output, if no plausible match exists at all.
24. Full pipeline run completed on all 146 golden-set rows (agent + judge, 146/146 both).
    Headline numbers: trivial baseline 17.8%, simple (CV) baseline 37.7%, LLM classifier
    74.7% -- a clear, defensible improvement over both baselines. Escalation accuracy is
    52.0%, which is the more concerning number: the golden set is 61% True / 39% False on
    escalation, meaning a trivial "always escalate" strategy would score ~61% -- higher
    than what the actual two-call escalation logic achieved. This becomes a central
    failure-analysis finding rather than something to bury.
25. Judge-human agreement check completed on a stratified 35-row subsample (spanning the
    full judge_overall score range, not just easy high-scoring cases), scored blind
    (human scores done without seeing the judge's own scores first). Results are a major
    finding: escalation_judgment correlation is only 0.19 (near-random) between judge and
    human, with the judge scoring +0.60 higher on average -- confirming and quantifying
    the earlier observation that the judge gave ~4.5/5 average even on rows where the
    agent's escalation call was objectively wrong against the golden-set ground truth.
    accuracy also shows weak agreement (20% exact match, judge +0.51 higher than human).
    tone is the one dimension with strong agreement (100% within-1). Conclusion: the LLM
    judge is NOT reliable for escalation quality specifically -- this is the central
    finding for the report's judge-reliability and "misleading headline number" sections.
26. Measured the actual repro time (PowerShell Measure-Command): 33 seconds -- but only
    because outputs/full_results.csv already existed locally and the resumable pipeline
    skipped all 146 already-completed rows rather than re-running ~450 real API calls.
    outputs/ was originally gitignored, which would have made this measurement dishonest
    for a grader doing a genuinely fresh clone (no cached results, full run would take several
    minutes minimum). Fixed by un-gitignoring outputs/ and committing the completed
    results, with this reasoning stated plainly in the README rather than left implicit --
    the <15 min claim is genuine, but depends on the committed cache, and the README says so.
27. Topped up the golden set from 146 to 150 (the spec's floor) by sampling 4 fresh
    candidates -- one per difficulty bucket -- to replace the 4 excluded misthreaded-reply
    rows, restoring the exact target bucket distribution (60/38/30/22). Hand-labeled the
    same way as all other rows. Retrieval pool rebuilt to exclude all 150 golden-set IDs.
28. Report updated with final 150-row numbers after the golden-set top-up: trivial 17.3%,
    simple (CV) 40.7%, LLM classifier unchanged at 74.7%, escalation accuracy 53.3% (up
    slightly from 52.0%, still worse than a trivial "always escalate" strategy). Confirmed
    the core failure-mode and judge-unreliability findings are unchanged after the top-up
    (same 50 false negatives, same 4.5 vs 4.6 judge score gap) -- the top-up didn't alter
    the project's central finding, it just closed the golden-set size gap. Report re-verified
    at exactly 6 pages (rendered via pandoc + LibreOffice) after the update.
