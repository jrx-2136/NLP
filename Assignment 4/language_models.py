"""Train and evaluate Laplace-smoothed n-gram language models."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


START = "<s>"
END = "</s>"
UNKNOWN = "<UNK>"
REQUIRED_SENTENCES = 100_000
DEV_SIZE = 1_000
TEST_SIZE = 1_000


class NGramModel:
    """An n-gram language model with add-one/Laplace smoothing."""

    def __init__(self, order: int, sentences: Iterable[list[str]]) -> None:
        self.order = order
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.vocabulary = {UNKNOWN, END}

        prepared_sentences = []
        for sentence in sentences:
            normalized = [token if token else UNKNOWN for token in sentence]
            self.vocabulary.update(normalized)
            prepared_sentences.append(normalized)

        self.vocabulary_size = len(self.vocabulary)
        self.token_count = 0
        for sentence in prepared_sentences:
            padded = [START] * (order - 1) + sentence + [END]
            self.token_count += len(sentence) + 1
            for index in range(order - 1, len(padded)):
                ngram = tuple(padded[index - order + 1 : index + 1])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def log_probability(self, token: str, context: tuple[str, ...]) -> float:
        if token not in self.vocabulary:
            token = UNKNOWN
        if self.order == 1:
            context = ()
        else:
            context = tuple(
                item if item in self.vocabulary or item == START else UNKNOWN
                for item in context[-(self.order - 1) :]
            )
        ngram = context + (token,)
        numerator = self.ngram_counts[ngram] + 1
        denominator = self.context_counts[context] + self.vocabulary_size
        return math.log(numerator / denominator)

    def sentence_log_probability(self, sentence: list[str]) -> tuple[float, int]:
        tokens = [token if token in self.vocabulary else UNKNOWN for token in sentence]
        padded = [START] * (self.order - 1) + tokens + [END]
        log_probability = 0.0
        predicted_tokens = 0
        for index in range(self.order - 1, len(padded)):
            context = tuple(padded[max(0, index - self.order + 1) : index])
            log_probability += self.log_probability(padded[index], context)
            predicted_tokens += 1
        return log_probability, predicted_tokens

    def perplexity(self, sentences: Iterable[list[str]]) -> float:
        total_log_probability = 0.0
        total_tokens = 0
        for sentence in sentences:
            sentence_log_probability, sentence_tokens = self.sentence_log_probability(sentence)
            total_log_probability += sentence_log_probability
            total_tokens += sentence_tokens
        if total_tokens == 0:
            return float("inf")
        return math.exp(-total_log_probability / total_tokens)


def load_sentences(input_path: Path) -> list[list[str]]:
    data = pd.read_parquet(input_path)
    if "tokenized_sentence" not in data.columns:
        raise ValueError("Input Parquet must contain a 'tokenized_sentence' column.")
    return [str(value).split() for value in data["tokenized_sentence"] if str(value).strip()]


def split_sentences(
    sentences: list[list[str]], seed: int, allow_smaller: bool
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    required = DEV_SIZE + TEST_SIZE + 1
    if len(sentences) < REQUIRED_SENTENCES and not allow_smaller:
        raise ValueError(
            f"Found {len(sentences):,} sentences, but at least {REQUIRED_SENTENCES:,} "
            "are required. Run Assignment 1 on a larger corpus or pass --allow-smaller "
            "only for a local smoke test."
        )
    if len(sentences) < required:
        raise ValueError(f"At least {required} sentences are needed to create the splits.")

    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    test = shuffled[:TEST_SIZE]
    development = shuffled[TEST_SIZE : TEST_SIZE + DEV_SIZE]
    training = shuffled[TEST_SIZE + DEV_SIZE :]
    return training, development, test


def save_model(model: NGramModel, output_dir: Path) -> None:
    with (output_dir / f"{model.order}gram_model.pkl").open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    assignment_dir = Path(__file__).resolve().parent
    default_input = assignment_dir.parent / "Assignment 1" / "indiccorp_tokenized.parquet"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=assignment_dir / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-smaller", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(input_path)
    training, development, test = split_sentences(sentences, args.seed, args.allow_smaller)
    splits = {"training": training, "development": development, "test": test}

    with (output_dir / "splits.json").open("w", encoding="utf-8") as file:
        json.dump(
            {name: len(values) for name, values in splits.items()},
            file,
            indent=2,
        )

    results = {
        "input_file": str(input_path),
        "total_sentences": len(sentences),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "seed": args.seed,
        "smoothing": "add-one/Laplace",
        "models": {},
    }

    for order in range(1, 5):
        model = NGramModel(order, training)
        save_model(model, output_dir)
        results["models"][f"{order}-gram"] = {
            "order": order,
            "vocabulary_size": model.vocabulary_size,
            "training_perplexity": model.perplexity(training),
            "development_perplexity": model.perplexity(development),
            "test_perplexity": model.perplexity(test),
        }
        print(
            f"{order}-gram: "
            f"train={results['models'][f'{order}-gram']['training_perplexity']:.4f}, "
            f"dev={results['models'][f'{order}-gram']['development_perplexity']:.4f}, "
            f"test={results['models'][f'{order}-gram']['test_perplexity']:.4f}"
        )

    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Saved models and results to {output_dir}")


if __name__ == "__main__":
    main()
