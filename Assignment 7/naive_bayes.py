"""Question 1: Naive Bayes Hindi POS tagger using the required features."""
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from pos_utils import DATA_DIR, read_tagged_file, evaluate_split, plot_tag_f1

OUT = Path(__file__).resolve().parent / "results_nb"


def make_features(sentence, i, previous_tag=None):
    word = sentence[i][0]
    prev_word = sentence[i - 1][0] if i > 0 else "<START>"
    next_word = sentence[i + 1][0] if i + 1 < len(sentence) else "<END>"
    # During training use the gold previous tag. During prediction use the
    # tag predicted for the preceding word (greedy left-to-right decoding).
    prev_tag = previous_tag if previous_tag is not None else (sentence[i - 1][1] if i > 0 else "<START>")
    return {
        "length": "MORE" if len(word) > 4 else "LESS",
        "word": word,
        "previous_word": prev_word,
        "next_word": next_word,
        "previous_tag": prev_tag,
    }


def training_examples(sentences):
    X, y = [], []
    for sent in sentences:
        for i, (_, tag) in enumerate(sent):
            X.append(make_features(sent, i))
            y.append(tag)
    return X, y


def predict_sentences(model, vectorizer, sentences):
    predictions = []
    for sent in sentences:
        pred_sent = []
        for i, (word, _) in enumerate(sent):
            features = make_features(sent, i, previous_tag=pred_sent[-1] if pred_sent else "<START>")
            pred_sent.append(model.predict(vectorizer.transform([features]))[0])
        predictions.append(pred_sent)
    return [tag for sent in predictions for tag in sent]


def main():
    train = read_tagged_file(DATA_DIR / "hindi-train-pos.txt")
    dev = read_tagged_file(DATA_DIR / "hindi-dev-pos.txt")
    test = read_tagged_file(DATA_DIR / "hindi-test-pos.txt")
    X_train, y_train = training_examples(train)
    vectorizer = DictVectorizer(sparse=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)
    labels = sorted(set(y_train))
    OUT.mkdir(exist_ok=True)
    results = {}
    for name, data in [("Dev", dev), ("Test", test)]:
        y_true = [tag for sent in data for _, tag in sent]
        y_pred = predict_sentences(model, vectorizer, data)
        results[name] = evaluate_split(name, y_true, y_pred, OUT, labels)
    plot_tag_f1({"Dev": results["Dev"], "Test": results["Test"]}, "Naive Bayes: per-tag F1", OUT / "nb_tag_f1.png")
    print(f"\nResults saved in: {OUT}")

if __name__ == "__main__":
    main()
