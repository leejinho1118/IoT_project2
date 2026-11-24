# gate.py notes
v1 (single version, no fixes needed): NFKC, ligature map, unicode quote/dash unify, de-hyphenate line breaks, lowercase, strip ALL whitespace and hyphens on both quote and PDF text. pypdf 6.16.1 text extraction.
False-positive review: the 9 failures are equation/math-notation quotes where the model wrote LaTeX/caret forms (e.g. f(x^k), $v_{b+1,k}$) or changed symbols (λ_i vs λl) versus PDF text layout; judged genuine non-verbatim, not normalization gaps. 2301.09604 raw JSON had a control char; parsed with json strict=False.
