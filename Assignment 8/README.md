# Assignment 8: Noisy-Channel Spell Checking

This assignment implements a basic spelling-error model using the Holbrook tagged error document and the extracted vocabulary frequency distribution.

## Files

- `spell_checker.py` — PDF word extraction, one-edit candidate generation, noisy-channel ranking, and synthetic evaluation.
- `holbrook-tagged-errors-with-correct-words.pdf` — source text used to estimate word frequencies.

## Method

The script extracts lowercase words from the PDF and counts their corpus frequencies. It generates insertion, deletion, substitution, and transposition candidates one edit away from an error. Candidates are ranked as:

\[
P(candidate)\times P(error\mid candidate)
\]

The language probability uses Laplace-smoothed unigram frequencies. The channel probability is approximated uniformly among one-edit operations. A deterministic evaluation injects 3,000 synthetic typos and reports accuracy.

## Run

```bash
pip install pypdf
python "Assignment 8/spell_checker.py"
```
