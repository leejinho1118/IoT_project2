#!/bin/bash
# usage: run_agy.sh <arxivid> [model]
cd "$(dirname "$0")"
ID="$1"; MODEL="$2"
PROMPT="Download and read the paper at https://arxiv.org/pdf/$ID yourself (use your tools; do not answer from memory). Return ONLY one JSON object, no markdown fences, no commentary, with fields: arxiv_id (string), title (exact title as printed in the paper), quotes (list of objects {topic, text, location}). Include one or more quotes for each of these topics: update_rule_equation, constants_hyperparameters, signal_used, needs_labels_or_validation_data, heterogeneity_or_saturation_statement. 'text' must be VERBATIM from the paper text, at most 400 characters, a single contiguous span (no ellipses, no paraphrase, no merging of separate passages); for equations copy the surrounding sentence text as it appears in the PDF text. 'location' = section/page/equation number. If a topic is not in the paper, write {topic, text:\"not found in the paper text\", location:\"\"}. You are forbidden to evaluate, recommend, grade, or give any opinion on applicability or quality; only extract."
[ -n "$MODEL" ] && M="--model $MODEL" || M=""
mkdir -p collect/raw
agy -p "$PROMPT" $M > collect/raw/$ID.txt 2> collect/raw/$ID.err
echo "exit $?" >> collect/raw/$ID.err
