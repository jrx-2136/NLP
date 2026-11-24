"""
NLP Lab Assignment 9 — Hindi POS Tagging
Models: MEMM, CRF, Naive Bayes, HMM
Dataset: Hindi-POS-Data (train/dev/test files supplied in Lab-Assignment-9.zip)

Install:
    pip install numpy pandas scikit-learn sklearn-crfsuite matplotlib joblib

Run:
    python assignment9_solution.py --data_dir ./Hindi-POS-Data --output_dir ./assignment9_results

Input format:
    one token per line: WORD<TAB>POS_TAG
    blank line separates sentences.
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, f1_score
)
import joblib
import sklearn_crfsuite


# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------

def read_tagged_sentences(path):
    """Read blank-line-separated WORD<TAB>TAG data."""
    sentences, current = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word, tag = parts[0], parts[-1]
            current.append((word, tag))
    if current:
        sentences.append(current)
    return sentences


def validate_data(train, dev, test):
    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        if not data:
            raise ValueError(f"{name} split is empty. Check the dataset path.")
        tokens = sum(map(len, data))
        print(f"{name:>5}: {len(data):,} sentences, {tokens:,} tokens")
    train_tags = {tag for sent in train for _, tag in sent}
    for name, data in [("dev", dev), ("test", test)]:
        unseen = {tag for sent in data for _, tag in sent} - train_tags
        if unseen:
            print(f"WARNING: tags in {name} but not train: {sorted(unseen)}")


# ---------------------------------------------------------------------
# 2. ASSIGNMENT FEATURES (shared by MEMM and CRF)
# ---------------------------------------------------------------------

def token_features(words, i):
    """
    Exact assignment feature specification:
      - Prefixes of lengths 1,2,3,4 (NULL if unavailable)
      - Suffixes of lengths 1,2,3,4 (NULL if unavailable)
      - Previous/current/next word
      - Binary length feature: LESS for <=4, otherwise MORE
    Sentence boundaries use BOS/EOS.
    """
    word = words[i]
    feats = {
        "prefix_1": word[:1] if len(word) >= 1 else "NULL",
        "prefix_2": word[:2] if len(word) >= 2 else "NULL",
        "prefix_3": word[:3] if len(word) >= 3 else "NULL",
        "prefix_4": word[:4] if len(word) >= 4 else "NULL",
        "suffix_1": word[-1:] if len(word) >= 1 else "NULL",
        "suffix_2": word[-2:] if len(word) >= 2 else "NULL",
        "suffix_3": word[-3:] if len(word) >= 3 else "NULL",
        "suffix_4": word[-4:] if len(word) >= 4 else "NULL",
        "previous_word": words[i - 1] if i > 0 else "BOS",
        "current_word": word,
        "next_word": words[i + 1] if i < len(words) - 1 else "EOS",
        "length_feature": "LESS" if len(word) <= 4 else "MORE",
    }
    return feats


def sentence_feature_dicts(words):
    return [token_features(words, i) for i in range(len(words))]


# ---------------------------------------------------------------------
# 3. EVALUATION AND REPORTING
# ---------------------------------------------------------------------

def flatten_gold(sentences):
    return [tag for sent in sentences for _, tag in sent]


def flatten_pred(predictions):
    return [tag for sent in predictions for tag in sent]


def evaluate_predictions(model_name, split_name, gold_sents, pred_sents, out_dir):
    y_true = flatten_gold(gold_sents)
    y_pred = flatten_pred(pred_sents)
    if len(y_true) != len(y_pred):
        raise ValueError(f"{model_name}/{split_name}: token count mismatch")
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    per_tag = pd.DataFrame({
        "tag": labels, "precision": p, "recall": r, "f1": f, "support": support
    })
    per_tag.insert(0, "model", model_name)
    per_tag.insert(1, "split", split_name)
    per_tag.to_csv(out_dir / f"{model_name}_{split_name}_per_tag.csv", index=False)

    summary = {
        "model": model_name, "split": split_name, "accuracy": float(accuracy),
        "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "tokens": len(y_true)
    }
    print(f"\n{model_name} — {split_name}")
    print(f"Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f} | "
          f"Weighted-F1: {weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    return summary, per_tag


# ---------------------------------------------------------------------
# 4. MEMM — MAXIMUM ENTROPY MARKOV MODEL
# ---------------------------------------------------------------------

class MEMMTagger:
    """
    A MEMM is a maximum-entropy (multinomial logistic-regression) model
    P(tag_t | word/context features, tag_(t-1)).
    Training uses the gold previous tag; decoding conditions on the
    previously predicted tag (greedy left-to-right decoding).

    This is a Python 3 compatible implementation of the sequential
    maximum-entropy POS-tagging approach in the assignment's referenced
    nltk-maxent-pos-tagger repository. The repository's original code
    targets old Python 2/NLTK APIs and hard-codes an external MEGAM binary.
    """

    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = LogisticRegression(
            solver="lbfgs", max_iter=120, C=2.0,
            class_weight=None, n_jobs=None, verbose=0
        )

    def _training_instances(self, sentences):
        X, y = [], []
        for sent in sentences:
            words = [w for w, _ in sent]
            gold_tags = [t for _, t in sent]
            for i, tag in enumerate(gold_tags):
                feat = token_features(words, i)
                feat["previous_tag"] = gold_tags[i - 1] if i > 0 else "BOS_TAG"
                X.append(feat)
                y.append(tag)
        return X, y

    def fit(self, sentences):
        X, y = self._training_instances(sentences)
        print(f"MEMM training instances: {len(y):,}")
        Xv = self.vectorizer.fit_transform(X)
        self.model.fit(Xv, y)
        return self

    def predict_sentence(self, words):
        result = []
        for i in range(len(words)):
            feat = token_features(words, i)
            feat["previous_tag"] = result[-1] if result else "BOS_TAG"
            Xv = self.vectorizer.transform([feat])
            result.append(self.model.predict(Xv)[0])
        return result

    def predict(self, sentences):
        return [self.predict_sentence([w for w, _ in s]) for s in sentences]


# ---------------------------------------------------------------------
# 5. CRF
# ---------------------------------------------------------------------

class CRFTagger:
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.05, c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
            verbose=True
        )

    def fit(self, sentences):
        X = [sentence_feature_dicts([w for w, _ in s]) for s in sentences]
        y = [[tag for _, tag in s] for s in sentences]
        self.model.fit(X, y)
        return self

    def predict(self, sentences):
        X = [sentence_feature_dicts([w for w, _ in s]) for s in sentences]
        return self.model.predict(X)


# ---------------------------------------------------------------------
# 6. NAIVE BAYES BASELINE
# ---------------------------------------------------------------------

class NaiveBayesTagger:
    """
    Sequential NB baseline. Uses the same word/context/morphology features
    and previous-tag feature as MEMM. Gold previous tags are used in
    training; predicted previous tags are used during decoding.
    """
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = MultinomialNB(alpha=0.1)

    def fit(self, sentences):
        X, y = [], []
        for sent in sentences:
            words = [w for w, _ in sent]
            tags = [t for _, t in sent]
            for i, tag in enumerate(tags):
                feat = token_features(words, i)
                feat["previous_tag"] = tags[i - 1] if i else "BOS_TAG"
                X.append(feat)
                y.append(tag)
        Xv = self.vectorizer.fit_transform(X)
        self.model.fit(Xv, y)
        return self

    def predict_sentence(self, words):
        pred = []
        for i in range(len(words)):
            feat = token_features(words, i)
            feat["previous_tag"] = pred[-1] if pred else "BOS_TAG"
            pred.append(self.model.predict(self.vectorizer.transform([feat]))[0])
        return pred

    def predict(self, sentences):
        return [self.predict_sentence([w for w, _ in s]) for s in sentences]


# ---------------------------------------------------------------------
# 7. HMM BASELINE — ADD-ONE SMOOTHING + VITERBI
# ---------------------------------------------------------------------

class HMMTagger:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.tags = []
        self.vocab = set()
        self.tag_counts = Counter()
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.start_counts = Counter()

    def fit(self, sentences):
        for sent in sentences:
            prev = "<START>"
            for word, tag in sent:
                self.vocab.add(word)
                self.tag_counts[tag] += 1
                self.emission_counts[tag][word] += 1
                self.transition_counts[prev][tag] += 1
                prev = tag
            self.transition_counts[prev]["<END>"] += 1
            self.start_counts[sent[0][1]] += 1
        self.tags = sorted(self.tag_counts)
        self.vocab.add("<UNK>")
        self.vocab_size = len(self.vocab)
        self.tag_size = len(self.tags)
        self.total_sentences = len(sentences)
        return self

    def _log_transition(self, prev, tag):
        count = self.transition_counts[prev][tag]
        denom = sum(self.transition_counts[prev].values()) + self.alpha * (self.tag_size + 1)
        return np.log((count + self.alpha) / denom)

    def _log_emission(self, tag, word):
        count = self.emission_counts[tag][word]
        denom = self.tag_counts[tag] + self.alpha * self.vocab_size
        return np.log((count + self.alpha) / denom)

    def predict_sentence(self, words):
        if not words:
            return []
        # Viterbi in log space
        dp = [{}]
        back = [{}]
        for tag in self.tags:
            dp[0][tag] = self._log_transition("<START>", tag) + self._log_emission(tag, words[0])
            back[0][tag] = None
        for i in range(1, len(words)):
            dp.append({})
            back.append({})
            for tag in self.tags:
                best_prev, best_score = None, -np.inf
                emit = self._log_emission(tag, words[i])
                for prev in self.tags:
                    score = dp[i - 1][prev] + self._log_transition(prev, tag) + emit
                    if score > best_score:
                        best_prev, best_score = prev, score
                dp[i][tag] = best_score
                back[i][tag] = best_prev
        last_tag = max(
            self.tags,
            key=lambda t: dp[-1][t] + self._log_transition(t, "<END>")
        )
        result = [last_tag]
        for i in range(len(words) - 1, 0, -1):
            result.append(back[i][result[-1]])
        return list(reversed(result))

    def predict(self, sentences):
        return [self.predict_sentence([w for w, _ in s]) for s in sentences]


# ---------------------------------------------------------------------
# 8. MAIN EXPERIMENT
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./Hindi-POS-Data")
    parser.add_argument("--output_dir", default="./assignment9_results")
    parser.add_argument("--skip_memm", action="store_true",
                        help="Skip MEMM if training time/resources are limited.")
    parser.add_argument("--skip_crf", action="store_true",
                        help="Skip CRF if training time/resources are limited.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train = read_tagged_sentences(data_dir / "hindi-train-pos.txt")
    dev = read_tagged_sentences(data_dir / "hindi-dev-pos.txt")
    test = read_tagged_sentences(data_dir / "hindi-test-pos.txt")
    validate_data(train, dev, test)

    models = {}
    if not args.skip_memm:
        models["MEMM"] = MEMMTagger()
    if not args.skip_crf:
        models["CRF"] = CRFTagger()
    models["NaiveBayes"] = NaiveBayesTagger()
    models["HMM"] = HMMTagger(alpha=0.1)

    summaries, per_tag_frames = [], []
    for name, model in models.items():
        print("\n" + "=" * 72)
        print(f"Training {name}")
        start = time.time()
        model.fit(train)
        print(f"{name} training time: {(time.time()-start)/60:.2f} minutes")
        joblib.dump(model, out_dir / f"{name.lower()}_model.joblib")

        for split_name, split in [("dev", dev), ("test", test)]:
            print(f"Predicting {split_name}...")
            predictions = model.predict(split)
            summary, per_tag = evaluate_predictions(
                name, split_name, split, predictions, out_dir
            )
            summaries.append(summary)
            per_tag_frames.append(per_tag)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "overall_comparison.csv", index=False)
    all_per_tag = pd.concat(per_tag_frames, ignore_index=True)
    all_per_tag.to_csv(out_dir / "all_models_per_tag_comparison.csv", index=False)

    # Overall comparison charts (test split)
    test_summary = summary_df[summary_df["split"] == "test"]
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        plt.figure(figsize=(9, 5))
        plt.bar(test_summary["model"], test_summary[metric])
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Model Comparison on Test Set — {metric.replace('_', ' ').title()}")
        plt.ylim(0, 1)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(out_dir / f"comparison_{metric}.png", dpi=200)
        plt.close()

    # Per-tag F1 chart: one chart per split, model grouped for each tag
    for split_name in ["dev", "test"]:
        subset = all_per_tag[all_per_tag["split"] == split_name]
        pivot = subset.pivot(index="tag", columns="model", values="f1").fillna(0)
        ax = pivot.plot(kind="bar", figsize=(max(14, len(pivot) * 0.55), 7))
        ax.set_title(f"Per-tag F1 Comparison — {split_name.upper()} Set")
        ax.set_xlabel("POS Tag")
        ax.set_ylabel("F1-score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"per_tag_f1_{split_name}.png", dpi=200)
        plt.close()

    # Per-tag accuracy is not a standard single-label metric; per-class
    # precision/recall/F1/support are reported instead, as required for
    # meaningful tag-wise comparison.
    with open(out_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print("\nCompleted. Results saved in:", out_dir.resolve())
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
