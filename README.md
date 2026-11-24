# NLP Assignments and Experiments

This repository contains Natural Language Processing laboratory assignments covering text preprocessing, word segmentation, finite-state methods, language models, smoothing, sequence labeling, spell checking, and neural word embeddings.

## Repository layout

- `Assignment 1/` — Marathi sentence/word tokenization and corpus statistics.
- `Assignment 2/` — greedy and dynamic-programming word segmentation.
- `Assignment 3/` — finite-state automaton and noun transducer exercises.
- `Assignment 4/` — unigram through quadrigram language models.
- `Assignment 5/` — add-k smoothing evaluation.
- `Assignment 6/` — interpolation and advanced smoothing methods.
- `Assignment 7/` — HMM and Naive Bayes Hindi POS tagging.
- `Assignment 8/` — noisy-channel spell checking.
- `Assignment 9/` — comparison of HMM, Naive Bayes, MEMM, and CRF POS taggers.
- `Assignment 10/` — Amazon review embeddings and neural network experiments.
- `nlp project/` — CRF POS tagging on the Universal Dependencies English-EWT data.

## General usage

Run assignment scripts from the repository root unless a README specifies otherwise. Examples use relative paths, so starting outside the repository may cause file-not-found errors. Most assignments require Python 3 and packages such as NumPy, pandas, scikit-learn, matplotlib, and PyTorch where noted.

## Generated and large files

The `.gitignore` excludes caches, virtual environments, large datasets, archives, serialized models, CSV data, and Parquet files. Large artifacts remain available locally but should not be committed unless deliberate external storage or Git LFS is used.
