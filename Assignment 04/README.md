# Assignment 4: N-gram Language Models

This assignment trains four Marathi language models from the tokenized output created by Assignment 1:

- Unigram
- Bigram
- Trigram
- Quadrigram (four-gram)

Each model uses add-one/Laplace smoothing:

\[
P(w_n \mid w_1, \ldots, w_{n-1}) =
\frac{C(w_1, \ldots, w_n) + 1}
{C(w_1, \ldots, w_{n-1}) + V}
\]

## Input and splits

By default, the program reads:

`../Assignment 1/indiccorp_tokenized.parquet`

The input must contain a `tokenized_sentence` column. Sentences are shuffled with a fixed seed, then split into:

- 1,000 test sentences
- 1,000 development sentences
- all remaining sentences for training

The current Assignment 1 artifact contains 1,000,000 sentences, so the required split contains 998,000 training sentences, 1,000 development sentences, and 1,000 test sentences.

## Run

From the repository root:

```bash
python "Assignment 4/language_models.py"
```

For a local smoke test with a smaller corpus only:

```bash
python "Assignment 4/language_models.py" --allow-smaller
```

Outputs are saved in `Assignment 4/outputs/`:

- `1gram_model.pkl` through `4gram_model.pkl`: trained model artifacts
- `splits.json`: split sizes
- `results.json`: vocabulary sizes and training/development/test perplexities

Use a different input or output directory with `--input` and `--output-dir`.
