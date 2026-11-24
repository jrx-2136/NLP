"""Question 2: HMM POS tagger with forward/backward Viterbi and forward-backward decoding."""
from collections import Counter, defaultdict
from math import log
from pathlib import Path
from pos_utils import DATA_DIR, read_tagged_file, evaluate_split, plot_tag_f1

OUT = Path(__file__).resolve().parent / "results_hmm"
START = "<START>"
END = "<END>"
UNK = "<UNK>"


class HMMTagger:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.tags = []
        self.vocab = set()
        self.log_trans = {}
        self.log_emit = {}
        self.log_start = {}
        self.log_end = {}

    def fit(self, sentences):
        tag_counts = Counter()
        emission_counts = Counter()
        transition_counts = Counter()
        start_counts = Counter()
        end_counts = Counter()
        vocab = set()
        for sent in sentences:
            if not sent:
                continue
            words = [w for w, _ in sent]
            tags = [t for _, t in sent]
            vocab.update(words)
            tag_counts.update(tags)
            start_counts[tags[0]] += 1
            end_counts[tags[-1]] += 1
            for word, tag in sent:
                emission_counts[tag, word] += 1
            previous = START
            for tag in tags:
                transition_counts[previous, tag] += 1
                previous = tag
            transition_counts[previous, END] += 1
        self.tags = sorted(tag_counts)
        self.vocab = vocab
        self.vocab.add(UNK)
        a = self.alpha
        n_tags = len(self.tags)
        n_vocab = len(self.vocab)
        total_sentences = sum(start_counts.values())
        self.log_start = {t: log((start_counts[t] + a) / (total_sentences + a * n_tags)) for t in self.tags}
        self.log_end = {}
        for t in self.tags:
            self.log_end[t] = log((end_counts[t] + a) / (tag_counts[t] + 2 * a))
        self.log_trans = {}
        for prev in [START] + self.tags:
            # All tags plus END are possible destinations.
            destinations = self.tags + [END]
            denom = sum(transition_counts[prev, dest] for dest in destinations) + a * len(destinations)
            for dest in destinations:
                self.log_trans[prev, dest] = log((transition_counts[prev, dest] + a) / denom)
        self.log_emit = {}
        for tag in self.tags:
            denom = tag_counts[tag] + a * n_vocab
            for word in self.vocab:
                self.log_emit[tag, word] = log((emission_counts[tag, word] + a) / denom)
        return self

    def emit(self, tag, word):
        return self.log_emit.get((tag, word if word in self.vocab else UNK), self.log_emit[tag, UNK])

    def forward_viterbi(self, words):
        """Standard left-to-right Viterbi dynamic programming."""
        if not words:
            return []
        dp, back = [], []
        dp.append({t: self.log_start[t] + self.emit(t, words[0]) for t in self.tags})
        back.append({t: None for t in self.tags})
        for i in range(1, len(words)):
            row, back_row = {}, {}
            for tag in self.tags:
                prev = max(self.tags, key=lambda p: dp[i-1][p] + self.log_trans[p, tag])
                row[tag] = dp[i-1][prev] + self.log_trans[prev, tag] + self.emit(tag, words[i])
                back_row[tag] = prev
            dp.append(row); back.append(back_row)
        last = max(self.tags, key=lambda t: dp[-1][t] + self.log_trans[t, END])
        path = [last]
        for i in range(len(words) - 1, 0, -1):
            path.append(back[i][path[-1]])
        return list(reversed(path))

    def backward_viterbi(self, words):
        """Right-to-left Viterbi recurrence; returns the best complete path."""
        n = len(words)
        if not n:
            return []
        # score[i][tag] is the best score for words i..end given tag at i.
        score = [{} for _ in range(n)]
        nxt = [{} for _ in range(n)]
        for tag in self.tags:
            score[n-1][tag] = self.emit(tag, words[n-1]) + self.log_trans[tag, END]
            nxt[n-1][tag] = None
        for i in range(n-2, -1, -1):
            for tag in self.tags:
                best_next = max(self.tags, key=lambda nt: self.log_trans[tag, nt] + self.emit(nt, words[i+1]) + score[i+1][nt])
                score[i][tag] = self.emit(tag, words[i]) + self.log_trans[tag, best_next] + score[i+1][best_next]
                nxt[i][tag] = best_next
        first = max(self.tags, key=lambda t: self.log_start[t] + score[0][t])
        path = [first]
        for i in range(n-1):
            path.append(nxt[i][path[-1]])
        return path

    def forward_backward(self, words):
        """Forward-backward posterior decoding: choose most probable tag at each position."""
        n = len(words)
        if not n:
            return []
        # Forward log probabilities alpha[i][tag]
        alpha = [{t: self.log_start[t] + self.emit(t, words[0]) for t in self.tags}]
        for i in range(1, n):
            row = {}
            for tag in self.tags:
                row[tag] = self.emit(tag, words[i]) + logsumexp([alpha[i-1][p] + self.log_trans[p, tag] for p in self.tags])
            alpha.append(row)
        # Backward log probabilities beta[i][tag]
        beta = [{} for _ in range(n)]
        beta[n-1] = {t: self.log_trans[t, END] for t in self.tags}
        for i in range(n-2, -1, -1):
            for tag in self.tags:
                beta[i][tag] = logsumexp([self.log_trans[tag, nt] + self.emit(nt, words[i+1]) + beta[i+1][nt] for nt in self.tags])
        return [max(self.tags, key=lambda t: alpha[i][t] + beta[i][t]) for i in range(n)]


def logsumexp(values):
    m = max(values)
    return m + log(sum(__import__("math").exp(v - m) for v in values))


def predict_all(model, sentences, algorithm):
    output = []
    for sent in sentences:
        words = [w for w, _ in sent]
        if algorithm == "Forward Viterbi":
            output.extend(model.forward_viterbi(words))
        elif algorithm == "Backward Viterbi":
            output.extend(model.backward_viterbi(words))
        else:
            output.extend(model.forward_backward(words))
    return output


def main():
    train = read_tagged_file(DATA_DIR / "hindi-train-pos.txt")
    dev = read_tagged_file(DATA_DIR / "hindi-dev-pos.txt")
    test = read_tagged_file(DATA_DIR / "hindi-test-pos.txt")
    model = HMMTagger(alpha=0.1).fit(train)
    OUT.mkdir(exist_ok=True)
    algorithms = ["Forward Viterbi", "Backward Viterbi", "Forward-Backward"]
    all_results = {}
    for split_name, data in [("Dev", dev), ("Test", test)]:
        y_true = [tag for sent in data for _, tag in sent]
        split_results = {}
        for algorithm in algorithms:
            y_pred = predict_all(model, data, algorithm)
            split_results[algorithm] = evaluate_split(f"{split_name} - {algorithm}", y_true, y_pred, OUT, model.tags)
        all_results[split_name] = split_results
        plot_tag_f1(split_results, f"HMM {split_name}: per-tag F1", OUT / f"hmm_{split_name.lower()}_tag_f1.png")
    print(f"\nResults saved in: {OUT}")

if __name__ == "__main__":
    main()
