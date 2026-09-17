# Hiver SDE Intern — AI Support Agent: Report

**Brand: XboxSupport** | Dataset: Customer Support on Twitter (Kaggle) | Golden set: 146 hand-labeled examples

---

## 1. Problem framing

### What "good" means for this brand

XboxSupport handles a narrow, single-ecosystem product surface (console hardware, games,
account, network) but at high volume with genuinely messy real-world traffic — typos, sarcasm,
missing context, multi-issue messages, and outright noise (spam, jokes, third-party mentions).
"Good" for this system is **not** "always produces a correct final answer." It is:

1. **Correctly triage the message** into one of 8 intents derived from real traffic (not
   guessed), so downstream handling (routing, drafting) starts from the right place.
2. **Draft a reply that is safe to show a human agent as a first pass** — grounded in how
   the brand has actually resolved similar issues before, not invented policy.
3. **Know when it doesn't know.** The escalation decision is the safety valve: when grounding
   is weak, the topic involves money or account security, or the message lacks enough context,
   the system should hand off to a human rather than guess.

Given that framing, (3) is arguably the most important property of the three, and — as this
report shows — the one where the system currently fails most.

### What we chose *not* to build

- **No fine-tuning.** All models are used zero-shot/few-shot via prompting (Groq API). Given
  the golden set is only 146 examples, fine-tuning would have no real train/test separation
  and no meaningful held-out signal.
- **No embedding-based retrieval.** Grounding uses TF-IDF (word-overlap) retrieval over real
  reconstructed (customer message, brand reply) pairs, not semantic embeddings. This was a
  deliberate timeline trade-off, and — per the failure analysis below — it's a real source of
  errors, not a cost-free simplification.
- **No multi-turn conversation handling.** Each message is scored independently; the system
  does not track conversation history across a thread, even though several golden-set messages
  are clearly mid-conversation fragments (see Failure Mode 4).
- **No non-English-specific handling.** One golden-set message is in French; it was labeled
  based on readable content, but the system makes no attempt to detect or specially handle
  non-English input.

---

## 2. Golden set

150 messages were sampled from XboxSupport's real customer traffic (21,276 candidate messages)
using a **composite difficulty score**, calibrated against real data percentiles rather than
guessed thresholds:

| Signal | Points | Threshold (calibrated against real data) |
|---|---|---|
| Matches 0 of 7 intent-keyword groups | +3 | — |
| Matches 2+ intent-keyword groups | +2 | — |
| Emotional marker | +1 | caps ratio > 30%, OR ≥1 exclamation mark, OR a frustration word |
| Unusual length | +1 | ≤5 words or ≥45 words |

Score → bucket: 0 = typical, 1 = edge, 2 = ambiguous, 3+ = adversarial. Target: 60/38/30/22
(typical/ambiguous/edge/adversarial), reduced from an initial 200-row target to 150 to make
genuinely careful hand-labeling achievable within the timeline. All 150 were labeled by hand:
intent, escalation decision + reason, and reply-quality notes, reading every message myself.

**4 messages were excluded** (not labeled) after discovering they were not customer messages
at all — text like *"Just so we're on the same page, what error code..."* signed with a
support-agent-style tag, revealing that the thread-reconstruction logic occasionally mis-threads
a brand-to-brand or chained reply as if it were a customer's. This recurred across all 4
labeling batches, confirming it's systematic, not a one-off — a real, documented data-quality
finding, kept as evidence rather than silently dropped.

**Final golden set: 146 rows.**

| Bucket | Count | | Intent | Count |
|---|---|---|---|---|
| typical | 59 | | other_unclear | 26 |
| ambiguous | 37 | | account_data_issue | 23 |
| edge | 29 | | network_connectivity | 23 |
| adversarial | 21 | | game_crash_bug | 20 |
| | | | hardware_issue | 19 |
| | | | how_to_question | 18 |
| | | | code_activation | 11 |
| | | | refund_request | 6 |

Escalation: 89 True / 57 False (61% of the golden set genuinely should escalate).

---

## 3. Results vs. baselines

| Method | Intent accuracy | Notes |
|---|---|---|
| **Trivial** (always guess most common intent) | **17.8%** | Floor. Majority class: other_unclear |
| **Simple** (TF-IDF + logistic regression, 5-fold CV) | **37.7%** | Cross-validated — never trained and tested on the same rows, since the golden set is the only labeled data available |
| **LLM classifier** (zero-shot) | **74.7%** | Clear, defensible improvement over both baselines |

The LLM classifier roughly **doubles** the simple baseline and **quadruples** the trivial floor.
This is a genuinely strong result for intent classification specifically.

**Escalation accuracy: 52.0%.** This is a serious problem, detailed in Section 4.

---

## 4. Failure analysis

### Failure Mode 1: Escalation is grounded in reply-pattern-matching, not topic-risk assessment

**The core finding of this project.** The confusion matrix on escalation:

| | Agent says: don't escalate | Agent says: escalate |
|---|---|---|
| **Should NOT escalate** | 37 | 20 |
| **Should escalate** | **50** | 39 |

50 cases where the agent should have escalated but didn't — 2.5x the 20 false-positive cases.
Reading the agent's own stated reasoning explains why: it repeatedly justifies *not* escalating
by saying the draft *"is consistent with grounding examples"* — even on a real refund request
("Feeling scammed," explicit refund ask), where the reasoning was: *"the past instances show a
clear pattern of not speculating on refunds... directing the user to live chat... without making
unauthorized commitments."* The agent is treating "my reply matches historical brand behavior"
as evidence that auto-sending is safe — but what the brand "usually says" to refund requests is
itself often just a deflection. Matching a deflection is not the same as correctly handling a
money-involving request that our own labeling standard says should always go to a human.

**Hypothesis:** the escalation prompt's criteria ("weak/contradictory grounding," "involves
money/security," "not confident") are being satisfied indirectly through the lens of *reply
consistency* rather than being checked directly against the message's actual content and stakes.

### Failure Mode 2: The judge does not catch Failure Mode 1

Average `judge_escalation_judgment` score on the 50 false-negative rows: **4.5/5** — nearly
identical to the overall average of 4.6/5. The judge is rating clearly wrong escalation
decisions as good, at scale. This is quantified precisely in Section 5.

### Failure Mode 3: Intent confusion between `game_crash_bug` and `hardware_issue`/`network_connectivity`

The largest misclassification pairs: `game_crash_bug` → `hardware_issue` (6 cases) and
`game_crash_bug` → `network_connectivity` (4 cases). Symptoms genuinely overlap — a game
"freezing" can look hardware-related, a game failing "online" can look network-related — and
the classifier doesn't reliably distinguish "the game/software is broken" from "the
console/network is broken."

### Failure Mode 4: Draft replies inherit irrelevant or hallucinated content from weak retrieval matches

Multiple low-judge-score rows show the draft pulling in content that doesn't belong: suggesting
an *Enforcement complaint* for a routine network disconnect (retrieved precedent was about
player conduct, similarity 0.26); a likely-hallucinated support-article URL; citing an unrelated
error code from a low-similarity match. TF-IDF (word-overlap) retrieval finds lexically similar
messages that aren't always actually the same situation, and the draft-generation step follows
weak grounding too literally instead of recognizing when grounding is too thin to use.

### Failure Mode 5: Draft content sometimes ignores information the customer already gave

One clean example: a customer explicitly states they're installing "from a **disc**"; the
draft's first question is *"Is this a disc or digital copy?"* — directly ignoring stated
information, which reads as inattentive and would erode trust if auto-sent.

---

## 5. Evaluation harness & judge reliability

LLM-as-judge (via Groq, `qwen/qwen3.8-27b`) scores each drafted reply on accuracy, relevance,
tone, escalation_judgment, and overall (1-5 each), against reply-quality notes written during
golden-set labeling.

**Judge-human agreement**, measured on a stratified 35-row subsample (spanning the full judge
score range, not just easy cases), scored **blind** — human scores were written without seeing
the judge's own scores first:

| Dimension | Exact match | Within 1 point | Correlation | Judge bias |
|---|---|---|---|---|
| accuracy | 20.0% | 62.9% | 0.28 | **+0.51** (judge scores higher) |
| relevance | 40.0% | 82.9% | 0.54 | +0.14 |
| tone | 42.9% | **100%** | 0.37 | +0.23 |
| **escalation_judgment** | 54.3% | 77.1% | **0.19** | **+0.60** |
| overall | 31.4% | 82.9% | 0.35 | +0.16 |

**`escalation_judgment`'s correlation with independent human scoring is 0.19 — close to
random — and the judge is systematically ~0.6 points more generous than a human.** Combined
with Failure Mode 2, this means **the judge cannot be trusted to catch the system's single
biggest weakness.** `tone` is the one dimension with strong agreement (100% within 1 point),
which makes sense — tone is lower-stakes and more subjective than correctness judgments.

---

## 6. What is misleading about my headline number?

If this report led with **"74.7% intent classification accuracy, clearly beating both
baselines"** — true, but dangerously incomplete, for several concrete reasons:

1. **The system's escalation accuracy (52.0%) is *worse* than a trivial "always escalate"
   strategy** (~61%, since 61% of the golden set genuinely needs escalation). A headline
   accuracy number for classification says nothing about this — and escalation is arguably
   the higher-stakes half of what this system does.

2. **The judge score that would normally reassure you about escalation quality
   (`escalation_judgment` averaging ~4.5-4.6/5) is itself unreliable** — 0.19 correlation with
   a human reviewer. Reporting that average without the agreement check would have hidden the
   exact failure this report is built around.

3. **The golden set's bucket composition is a deliberate design choice, not the true traffic
   distribution.** 40% of it (59/146) is "typical" — the easiest bucket by construction. Real
   production traffic's actual difficulty mix is unknown; if it skews harder than our sample,
   the 74.7% would not hold.

4. **A smaller, structural example from earlier in this project**: while selecting a brand,
   AmazonHelp showed a "2% deflection rate" by keyword search — looking like the most
   substantive brand available. Reading actual reply text showed most of its replies were
   deflections to an external form ("fill this form: [link]") that the keyword list simply
   didn't catch. A single misleading summary statistic, caught only by looking at raw data
   instead of trusting an aggregate number — the same lesson this whole report is built on,
   just at a different stage of the project.

---

## 7. Decision log

See `decision_log.md` — 25 entries covering every non-obvious choice, including several
real incidents (a reasoning-model token-truncation bug, two exhausted free-tier quota pools,
a wrong model ID, and a judge field-name typo) and how each was diagnosed and fixed, not just
the final state.

---

## 8. What I'd do with one more week

1. **Fix the escalation logic directly** — decouple "is this draft consistent with grounding"
   from "does this topic/situation warrant human review." The latter should be checked against
   the message's actual content (money, security, explicit escalation requests, stated prior
   failed attempts) independent of how well a draft happens to match retrieval precedent.
2. **Replace TF-IDF retrieval with embeddings** (e.g. sentence-transformers) and measure
   whether it actually reduces Failure Mode 4 — not assumed to be better, tested.
3. **Fix the systematic misthreaded-brand-reply bug** in thread reconstruction (Section 2)
   before it affects the retrieval pool's grounding quality, not just the golden set.
4. **Grow the golden set past 146** now that the labeling process and tooling are proven, to
   tighten the confidence intervals on every headline number in this report.
5. **Build a genuinely independent escalation-specific judge rubric**, informed by exactly
   which cases the human/judge disagreement subsample revealed as hardest.
