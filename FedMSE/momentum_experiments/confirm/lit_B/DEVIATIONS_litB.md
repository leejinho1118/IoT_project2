# Deviations (2026-11-24)
- COLLECTION role (normally agy) filled by codex research tier because agy quota exhausted on both pools (resets 45h/123h). Papers: `codex --search exec -m gpt-6-luna -s read-only` (network worked in read-only). Extra-candidate listing: gpt-6.1-terra. Per user rule, no sol. No Opus.
- Collectors forbidden from verdict/ranking/opinion (prompt copied from run_agy.sh). CRITERIA_litB.md untouched, sha256 verified OK before start.
- Helper run_codex.sh added; prompts in prompts/.
- 2312.12670, 2510.04203, 2505.11126 added from the terra list (extra_candidates); collected by codex gpt-6-luna with the same prompt/run_codex.sh (same deviation as above), gated with gate.py. pdf/<id>.txt = full pypdf-extracted text for judges.
