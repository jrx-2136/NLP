# Assignment 7: Hindi POS Tagging

This assignment trains and evaluates two part-of-speech tagging approaches on the supplied Hindi train, development, and test splits.

## Files

- `naive_bayes.py` — feature-based Multinomial Naive Bayes tagger.
- `hmm_tagger.py` — HMM implementation with forward and backward Viterbi decoding and forward-backward posterior decoding.
- `pos_utils.py` — dataset loading, evaluation, CSV reporting, and F1 chart generation.
- `Hindi-POS-data/` — tagged Hindi training, development, and test files.
- `results_nb/` — Naive Bayes per-tag reports and charts.
- `results_hmm/` — HMM per-algorithm reports and charts.

## Models

The Naive Bayes features include word length, the current and neighboring words, and the previous predicted tag. Training is not a simple independent-token exercise because the gold previous tag is used to create training examples; inference uses the preceding predicted tag.

The HMM uses add-alpha transition, start, end, and emission probabilities. Forward Viterbi and backward Viterbi find complete best tag sequences, while forward-backward chooses the highest-marginal-probability tag independently at each position.

## Run

```bash
pip install scikit-learn matplotlib
python "Assignment 7/naive_bayes.py"
python "Assignment 7/hmm_tagger.py"
```

Data lines use `word<TAB>POS_TAG`, with blank lines separating sentences.
