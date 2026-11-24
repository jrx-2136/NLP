# NLP Project: CRF POS Tagging

This project trains and evaluates a Conditional Random Field POS tagger on the Universal Dependencies English-EWT corpus.

## Files

- `crf-pos-tagger-same-ewt-dataset.ipynb` — end-to-end CRF experiment and analysis notebook.
- `crf_metrics.json` — saved test metrics and CRF hyperparameters.
- `crf_test_predictions.jsonl` — test-set predictions in one-JSON-object-per-line format.
- `UD_English-EWT/` — English-EWT CoNLL-U dataset and its release documentation/tools.

## Data and method

The notebook reads the `en_ewt-ud-train.conllu`, `en_ewt-ud-dev.conllu`, and `en_ewt-ud-test.conllu` splits. It extracts POS and context features from each token, trains a CRF sequence model, predicts the test sequence, and writes evaluation metrics and predictions.

The saved metric file is from a CRF run with test accuracy and macro-F1 and includes training, inference, and throughput measurements. These values should be interpreted as the results of that recorded run, not as guaranteed scores for a differently configured notebook run.

## Run

Open `crf-pos-tagger-same-ewt-dataset.ipynb` in Jupyter or VS Code and run it with the repository's Python environment. The notebook should be run from this folder or have notebook-relative paths resolved consistently.
