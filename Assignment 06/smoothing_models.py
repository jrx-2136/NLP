"""Evaluate Assignment 4 n-gram models with five smoothing techniques."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

ASSIGNMENT_DIR = Path(__file__).resolve().parent

START = "<s>"
END = "</s>"
UNKNOWN = "<UNK>"
REQUIRED_SENTENCES = 10_000
DEV_SIZE = 1_000
TEST_SIZE = 1_000


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
    return (
        shuffled[TEST_SIZE + DEV_SIZE :],
        shuffled[TEST_SIZE : TEST_SIZE + DEV_SIZE],
        shuffled[:TEST_SIZE],
    )


METHODS = (
    "interpolation",
    "good_turing",
    "katz_backoff",
    "stupid_backoff",
    "kneser_ney",
)


class CountStore:
    """Count n-grams from orders one through four in one training pass."""

    def __init__(self, sentences: list[list[str]], max_order: int = 4) -> None:
        self.max_order = max_order
        self.vocabulary = {UNKNOWN, END}
        self.ngrams: dict[int, Counter[tuple[str, ...]]] = {
            order: Counter() for order in range(1, max_order + 1)
        }
        self.contexts: dict[int, Counter[tuple[str, ...]]] = {
            order: Counter() for order in range(1, max_order + 1)
        }
        self.followers: dict[int, defaultdict[tuple[str, ...], set[str]]] = {
            order: defaultdict(set) for order in range(1, max_order + 1)
        }
        self.preceders: defaultdict[str, set[str]] = defaultdict(set)

        for sentence in sentences:
            tokens = [token if token else UNKNOWN for token in sentence]
            self.vocabulary.update(tokens)
            for order in range(1, max_order + 1):
                padded = [START] * (order - 1) + tokens + [END]
                for index in range(order - 1, len(padded)):
                    ngram = tuple(padded[index - order + 1 : index + 1])
                    context = ngram[:-1]
                    token = ngram[-1]
                    self.ngrams[order][ngram] += 1
                    self.contexts[order][context] += 1
                    self.followers[order][context].add(token)
                    if order == 2:
                        self.preceders[token].add(context[0])

        self.ngrams[1][(UNKNOWN,)] += 1
        self.contexts[1][()] += 1
        self.vocabulary_size = len(self.vocabulary)
        self.total_unigrams = self.contexts[1][()]
        self.discount = 0.75
        self._count_of_counts = {
            order: Counter(self.ngrams[order].values()) for order in range(1, max_order + 1)
        }

    def normalize(self, token: str) -> str:
        return token if token in self.vocabulary else UNKNOWN

    def context_for(self, order: int, context: tuple[str, ...]) -> tuple[str, ...]:
        if order == 1:
            return ()
        return tuple(self.normalize(item) if item != START else START for item in context[-order + 1 :])

    def ngram(self, order: int, context: tuple[str, ...], token: str) -> tuple[str, ...]:
        return self.context_for(order, context) + (self.normalize(token),)

    def mle(self, order: int, context: tuple[str, ...], token: str) -> float:
        ngram = self.ngram(order, context, token)
        context_key = ngram[:-1]
        denominator = self.contexts[order][context_key]
        if denominator == 0:
            return 0.0
        return self.ngrams[order][ngram] / denominator

    def adjusted_count(self, order: int, count: int) -> float:
        if count == 0:
            total = sum(self.ngrams[order].values())
            return max(self._count_of_counts[order][1] / max(total, 1), 1e-12)
        current = self._count_of_counts[order][count]
        following = self._count_of_counts[order][count + 1]
        if current == 0 or following == 0 or count > 5:
            return float(count)
        return (count + 1) * following / current

    @lru_cache(maxsize=None)
    def good_turing(self, order: int, context: tuple[str, ...], token: str) -> float:
        ngram = self.ngram(order, context, token)
        context_key = ngram[:-1]
        observed = self.contexts[order][context_key]
        if observed == 0:
            return 1.0 / self.vocabulary_size
        observed_tokens = self.followers[order][context_key]
        adjusted_sum = sum(
            self.adjusted_count(order, self.ngrams[order][context_key + (item,)])
            for item in observed_tokens
        )
        unseen_count = max(self.vocabulary_size - len(observed_tokens), 0)
        unseen_mass = self.adjusted_count(order, 0)
        denominator = adjusted_sum + unseen_count * unseen_mass
        if denominator == 0:
            return 1.0 / self.vocabulary_size
        return self.adjusted_count(order, self.ngrams[order][ngram]) / denominator

    @lru_cache(maxsize=None)
    def katz(self, order: int, context: tuple[str, ...], token: str) -> float:
        if order == 1:
            return self.good_turing(1, (), token)
        ngram = self.ngram(order, context, token)
        context_key = ngram[:-1]
        if self.ngrams[order][ngram] > 0:
            return self.good_turing(order, context_key, token)

        observed_tokens = self.followers[order][context_key]
        lower_context = context_key[1:]
        observed_total = sum(
            self.good_turing(order, context_key, candidate)
            for candidate in observed_tokens
        )
        # The lower-order Katz distribution is normalized, so its candidate
        # probabilities sum to one without enumerating the vocabulary.
        alpha = max(1.0 - observed_total, 1e-12)
        return alpha * self.katz(order - 1, lower_context, token)

    def stupid(self, order: int, context: tuple[str, ...], token: str) -> float:
        token = self.normalize(token)
        if order == 1:
            return max(self.mle(1, (), token), 1e-12)
        ngram = self.ngram(order, context, token)
        if self.ngrams[order][ngram] > 0:
            return self.mle(order, ngram[:-1], token)
        return 0.4 * self.stupid(order - 1, ngram[:-2], token)

    @lru_cache(maxsize=None)
    def kneser_ney(self, order: int, context: tuple[str, ...], token: str) -> float:
        token = self.normalize(token)
        if order == 1:
            total_continuations = sum(len(values) for values in self.preceders.values())
            probability = len(self.preceders[token]) / max(total_continuations, 1)
            return max(probability, 1e-12)

        ngram = self.ngram(order, context, token)
        context_key = ngram[:-1]
        context_count = self.contexts[order][context_key]
        if context_count == 0:
            return self.kneser_ney(order - 1, context_key[1:], token)
        followers = self.followers[order][context_key]
        discounted = max(self.ngrams[order][ngram] - self.discount, 0.0) / context_count
        backoff_weight = self.discount * len(followers) / context_count
        return discounted + backoff_weight * self.kneser_ney(order - 1, context_key[1:], token)

    def probability(self, method: str, order: int, context: tuple[str, ...], token: str) -> float:
        token = self.normalize(token)
        if method == "interpolation":
            probabilities = [self.mle(level, context[-level + 1 :], token) for level in range(1, order + 1)]
            return max(sum(probabilities) / order, 1e-12)
        if method == "good_turing":
            return max(self.good_turing(order, context, token), 1e-12)
        if method == "katz_backoff":
            return max(self.katz(order, context, token), 1e-12)
        if method == "stupid_backoff":
            return self.stupid(order, context, token)
        if method == "kneser_ney":
            return self.kneser_ney(order, context, token)
        raise ValueError(f"Unknown smoothing method: {method}")


def evaluate_method(store: CountStore, method: str, order: int, sentences: list[list[str]]) -> float:
    total_log_probability = 0.0
    total_tokens = 0
    for sentence in sentences:
        tokens = [store.normalize(token) for token in sentence]
        padded = [START] * (order - 1) + tokens + [END]
        for index in range(order - 1, len(padded)):
            context = tuple(padded[max(0, index - order + 1) : index])
            probability = store.probability(method, order, context, padded[index])
            total_log_probability += math.log(max(probability, 1e-12))
            total_tokens += 1
    if total_tokens == 0:
        return float("inf")
    return math.exp(-total_log_probability / total_tokens)


def main() -> None:
    default_input = ASSIGNMENT_DIR.parent / "Assignment 1" / "indiccorp_tokenized.parquet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=ASSIGNMENT_DIR / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-smaller", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    sentences = load_sentences(input_path)
    training, development, test = split_sentences(sentences, args.seed, args.allow_smaller)
    store = CountStore(training)
    results = {
        "input_file": str(input_path),
        "total_sentences": len(sentences),
        "split_sizes": {
            "training": len(training),
            "development": len(development),
            "test": len(test),
        },
        "seed": args.seed,
        "vocabulary_size": store.vocabulary_size,
        "methods": {},
    }

    for method in METHODS:
        results["methods"][method] = {}
        for order in range(1, 5):
            development_perplexity = evaluate_method(store, method, order, development)
            test_perplexity = evaluate_method(store, method, order, test)
            results["methods"][method][f"{order}-gram"] = {
                "order": order,
                "development_perplexity": development_perplexity,
                "test_perplexity": test_perplexity,
            }
            print(
                f"{method:16} {order}-gram: "
                f"dev={development_perplexity:.4f}, test={test_perplexity:.4f}"
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoothing_results.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
