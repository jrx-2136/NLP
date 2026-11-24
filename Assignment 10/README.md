# Assignment 10: Embeddings and Neural Networks

This assignment studies pretrained word embeddings and neural classifiers using Amazon review data.

## Files

- `NLP_Lab_Assignment_10_Solution.ipynb` — preprocessing, sentence-vector construction, analogy queries, and neural-network experiments.
- `nlp_lab10_u24ai014.pdf` — assignment specification or reference material.
- `amazon/` — Amazon review source files and optional downloaded archive.
- `archive (6).zip` — downloaded dataset archive.
- `data/` and `embeddings/` — intended locations for extracted review data and pretrained vectors.

## Questions

The notebook compares Google News Word2Vec, GloVe, and FastText representations. It constructs document or sentence vectors from word vectors, evaluates word analogies, and trains single-hidden-layer neural-network classifiers on review sentiment.

## Setup

Place `train.csv` and `test.csv` under `amazon/` or `data/amazon_reviews/` according to the notebook configuration. Add pretrained Word2Vec, GloVe, and FastText vectors under `embeddings/`. Word2Vec and GloVe files can require substantial memory and are expected to be downloaded separately.

Open the notebook in Jupyter or VS Code and run the cells sequentially. Load embeddings one at a time when working with limited RAM.
