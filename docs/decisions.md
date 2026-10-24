# Unspecified decisions

The brief did not tell you everything. Every time you or your agent chose
something the brief did not specify, add a line here.

Examples of the kind of thing that counts: what fields the record has, what
happens to an issue with no clear type, how confident is confident enough,
what the output looks like when the model returns nothing usable.

You will need this list in W2, when we reconstruct the requirements this
build implicitly assumed. The comparison across the cohort is the lesson, so
write down what you actually decided, not what you think is the right answer.

---

- Record fields: repo, number, url, issue_type, component, severity, has_reproduction_steps, summary, confidence, needs_human_review, review_reasons.
- issue_type vocabulary: bug / enhancement / documentation / question / other (union of the repos' gold types; hono's "meta" folded into "other").
- An issue with no clear type is labelled "other" and flagged for human review rather than forced into a type.
- component is restricted to the issue's own repo vocabulary from config/repos.toml; anything outside it becomes null and is flagged. Textual has no components, so it is always null.
- severity scale is critical/high/medium/low, defined in the prompt (crash/data loss/security = critical); there is no gold data for it.
- "Confident" means model-reported confidence >= 0.6; below that, needs_human_review is true. The number is the model's self-report, not calibrated.
- confidence measures certainty of issue_type only, not of severity or component.
- Invalid model output (unknown type/severity, non-numeric confidence) is coerced to a safe default (other / medium / 0.0) and flagged, never raised.
- A failed model call (network, rate limit, parse error) still prints a record: type "other", confidence 0, needs_human_review with the reason. The CLI does not exit non-zero.
- The model sees only repo, title and body. Existing GitHub labels, state, comments and gold fields are withheld (labels are the gold source, and new issues have none).
- Issue bodies are truncated to the first 6000 characters before being sent to the model.
- Issue text is treated as data: the prompt tells the model to ignore instructions inside it, and nothing from an issue ever reaches a shell.
- Summary is one model-written sentence, whitespace-collapsed and capped at 300 chars; it falls back to the title if empty.
- Output format is one JSON object per line (JSONL) on stdout, so batches can be piped to a router.
- Input: an issue from data/snapshot by --number/--index/--limit, or a JSON file in the snapshot schema. Live GitHub fetching is out of scope (lab says not to hit the API).
- The CLI refuses --split test because test.json is sealed.
- Model is whatever LLM_MODEL in .env names (Gemini here); default sampling settings, dspy.Predict (no chain-of-thought), 3 retries.
- Issue number is output as a string ("3994"), even though dev.json stores it as an integer.
- Default tests are offline (fake model); the live-model test is opt-in via TRIAGE_LIVE_TESTS=1 and skips, rather than fails, when the provider quota is exhausted.
- DSPy's on-disk response cache is left on, so re-triaging the same issue may not call the model again.
