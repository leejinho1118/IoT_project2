#!/bin/bash
cd "$(dirname "$0")"
I=$1; M=${2:-gpt-6-luna}; O=${3:-$I}
codex --search exec -m $M -s read-only -o collect/$O.raw.txt - < prompts/$I.txt > collect/$O.log 2>&1
