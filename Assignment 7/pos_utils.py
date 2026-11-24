"""Shared utilities for Hindi POS tagging Lab Assignment 7."""
from pathlib import Path
from collections import Counter
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Hindi-POS-Data"


def read_tagged_file(path):
    """Read word<TAB>tag data. Blank lines mark sentence boundaries."""
    sentences, sentence = [], []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.rsplit(None, 1)
            if len(parts) >= 2:
                word, tag = parts[0], parts[-1]
                sentence.append((word, tag))
    if sentence:
        sentences.append(sentence)
    return sentences


def flatten(sentences):
    words, tags = [], []
    for sent in sentences:
        for word, tag in sent:
            words.append(word)
            tags.append(tag)
    return words, tags


def evaluate_split(name, y_true, y_pred, output_dir, labels=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    print(f"\n{'=' * 18} {name} {'=' * 18}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4))
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    csv_path = output_dir / f"{name.lower().replace(' ', '_')}_per_tag_f1.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "precision", "recall", "f1-score", "support"])
        for tag in labels:
            r = report.get(tag, {})
            writer.writerow([tag, r.get("precision", 0), r.get("recall", 0), r.get("f1-score", 0), r.get("support", 0)])
    return {tag: report.get(tag, {}).get("f1-score", 0.0) for tag in labels}


def plot_tag_f1(results, title, path):
    """results maps algorithm name to {tag: F1}; plot grouped bars."""
    algorithms = list(results)
    tags = sorted(set().union(*(set(v) for v in results.values())))
    if not tags:
        return
    width = 0.8 / max(1, len(algorithms))
    x = list(range(len(tags)))
    plt.figure(figsize=(max(12, len(tags) * 0.45), 6))
    for i, algorithm in enumerate(algorithms):
        vals = [results[algorithm].get(tag, 0.0) for tag in tags]
        offsets = [v + (i - (len(algorithms) - 1) / 2) * width for v in x]
        plt.bar(offsets, vals, width=width, label=algorithm)
    plt.xticks(x, tags, rotation=70, ha="right")
    plt.ylabel("F1-score")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved chart: {path}")


def sentence_tag_sequences(sentences):
    return [[tag for _, tag in s] for s in sentences]
