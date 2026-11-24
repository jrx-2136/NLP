# Assignment 5: Add-k Smoothing

This assignment evaluates the unigram, bigram, trigram, and quadrigram models from Assignment 4 using Add-k smoothing. The default value is `k=0.3`.

For an n-gram and its context, the implementation uses:

\[
P(w_i \mid w_{i-N+1}^{i-1}) =
\frac{C(w_{i-N+1}^{i}) + k}
{C(w_{i-N+1}^{i-1}) + kV}
\]

where `V` is the training vocabulary size. Development and test perplexities are written to JSON.

## Run

From the repository root:

```bash
python "Assignment 5/add_k_language_models.py"
```

The script uses the same input, random seed, and 1,000-sentence development/test split as Assignment 4. Results are saved under `Assignment 5/outputs/` as `add_k_0.3_results.json`.

For a local smoke test with the currently available smaller Assignment 1 corpus:

```bash
python "Assignment 5/add_k_language_models.py" --allow-smaller
```

To test another positive value of `k`:

```bash
python "Assignment 5/add_k_language_models.py" --k 0.5 --allow-smaller
```

The default run enforces the minimum corpus size configured by Assignment 4. Run Assignment 1 on the required full corpus before collecting final assignment results.
