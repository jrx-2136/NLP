# Assignment 6: Smoothing Techniques

This assignment evaluates the four n-gram models from Assignment 4 using:

1. Interpolated smoothing
2. Good-Turing smoothing
3. Katz backoff smoothing
4. Stupid backoff smoothing
5. Interpolated Kneser-Ney smoothing

The models are evaluated on the same deterministic training, development, and test split used by Assignments 4 and 5. The implementation supports unigram, bigram, trigram, and quadrigram orders.

## Run

From the repository root:

```bash
python "Assignment 6/smoothing_models.py"
```

Results are written to `Assignment 6/outputs/smoothing_results.json`.

For a local smoke test with the currently available smaller corpus:

```bash
python "Assignment 6/smoothing_models.py" --allow-smaller
```

The default input is:

`../Assignment 1/indiccorp_tokenized.parquet`

Use another corpus with `--input path/to/file.parquet`, and choose a different output directory with `--output-dir path/to/output`.

## Implementation details

- Interpolation averages the maximum-order MLE and all lower-order MLE probabilities.
- Good-Turing uses frequency-of-frequency counts, with a conservative fallback for sparse counts.
- Katz backoff uses Good-Turing probabilities for observed n-grams and recursively backs off for unseen n-grams.
- Stupid backoff uses a `0.4` backoff multiplier.
- Interpolated Kneser-Ney uses absolute discounting with discount `0.75` and lower-order continuation probabilities.
- Unknown evaluation tokens are mapped to `<UNK>`.

Stupid backoff does not define a normalized probability distribution. Its reported exponentiated negative mean log score is therefore a **pseudo-perplexity**, while the other methods produce probability perplexities.

The full assignment requires at least 100,000 sentences. The current Assignment 1 artifact is smaller, so use `--allow-smaller` only for smoke testing until the corpus is regenerated.
