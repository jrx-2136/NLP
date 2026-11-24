"""Evaluate four n-gram models with Add-k smoothing."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

ASSIGNMENT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_4_DIR = ASSIGNMENT_DIR.parent / "Assignment 4"
sys.path.insert(0, str(ASSIGNMENT_4_DIR))

from language_models import END, UNKNOWN, START, load_sentences, split_sentences  # noqa: E402


class AddKModel:
    """An n-gram model using Add-k smoothing for a fixed k."""

    def __init__(self, order: int, sentences: Iterable[list[str]], k: float) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero.")
        self.order = order
        self.k = k
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.vocabulary = {UNKNOWN, END}

        prepared_sentences = []
        for sentence in sentences:
            normalized = [token if token else UNKNOWN for token in sentence]
            self.vocabulary.update(normalized)
            prepared_sentences.append(normalized)

        self.vocabulary_size = len(self.vocabulary)
        for sentence in prepared_sentences:
            padded = [START] * (order - 1) + sentence + [END]
            for index in range(order - 1, len(padded)):
                ngram = tuple(padded[index - order + 1 : index + 1])
                self.ngram_counts[ngram] += 1
                self.context_counts[ngram[:-1]] += 1

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
        numerator = self.ngram_counts[ngram] + self.k
        denominator = self.context_counts[context] + self.k * self.vocabulary_size
        return math.log(numerator / denominator)

    def sentence_log_probability(self, sentence: list[str]) -> tuple[float, int]:
        tokens = [token if token in self.vocabulary else UNKNOWN for token in sentence]
        padded = [START] * (self.order - 1) + tokens + [END]
        total_log_probability = 0.0
        predicted_tokens = 0
        for index in range(self.order - 1, len(padded)):
            context = tuple(padded[max(0, index - self.order + 1) : index])
            total_log_probability += self.log_probability(padded[index], context)
            predicted_tokens += 1
        return total_log_probability, predicted_tokens

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


def evaluate(input_path: Path, output_dir: Path, k: float, seed: int, allow_smaller: bool) -> dict:
    sentences = load_sentences(input_path)
    training, development, test = split_sentences(sentences, seed, allow_smaller)
    splits = {"training": training, "development": development, "test": test}

    results = {
        "input_file": str(input_path),
        "total_sentences": len(sentences),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "seed": seed,
        "smoothing": "Add-k",
        "k": k,
        "models": {},
    }

    for order in range(1, 5):
        model = AddKModel(order, training, k)
        results["models"][f"{order}-gram"] = {
            "order": order,
            "vocabulary_size": model.vocabulary_size,
            "development_perplexity": model.perplexity(development),
            "test_perplexity": model.perplexity(test),
        }
        metrics = results["models"][f"{order}-gram"]
        print(
            f"{order}-gram (k={k:g}): "
            f"dev={metrics['development_perplexity']:.4f}, "
            f"test={metrics['test_perplexity']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"add_k_{k:g}_results.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Saved results to {output_path}")
    return results


def main() -> None:
    default_input = ASSIGNMENT_DIR.parent / "Assignment 1" / "indiccorp_tokenized.parquet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=ASSIGNMENT_DIR / "outputs")
    parser.add_argument("--k", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-smaller", action="store_true")
    args = parser.parse_args()

    evaluate(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        k=args.k,
        seed=args.seed,
        allow_smaller=args.allow_smaller,
    )


if __name__ == "__main__":
    main()
