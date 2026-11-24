
import re
import random
import string
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

# ==========================================
# 1. CONFIGURATION
# ==========================================

PDF_FILE = Path(__file__).resolve().parent / "holbrook-tagged-errors-with-correct-words.pdf"
RANDOM_SEED = 42
TEST_SIZE = 3000


# ==========================================
# 2. DATASET PREPROCESSING
# ==========================================

def extract_words(pdf_file):
    """Extract words from the Holbrook PDF dataset."""

    reader = PdfReader(pdf_file)

    text = " ".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    words = re.findall(r"[a-z]+", text.lower())

    return words


# ==========================================
# 3. GENERATE EDIT DISTANCE 1 CANDIDATES
# ==========================================

def edits1(word):
    """
    Generate all words one edit away:
    1. Insertion
    2. Deletion
    3. Substitution
    4. Transposition
    """

    letters = string.ascii_lowercase

    splits = [
        (word[:i], word[i:])
        for i in range(len(word) + 1)
    ]

    result = set()

    for left, right in splits:

        # Deletion
        if right:
            result.add(left + right[1:])

        # Transposition
        if len(right) > 1:
            result.add(
                left + right[1] + right[0] + right[2:]
            )

        # Substitution
        if right:
            for ch in letters:
                result.add(left + ch + right[1:])

        # Insertion
        for ch in letters:
            result.add(left + ch + right)

    result.discard(word)

    return result


# ==========================================
# 4. NOISY CHANNEL SPELL CHECKER
# ==========================================

class NoisyChannelSpellChecker:

    def __init__(self, words):

        # Calculate unigram frequencies
        self.unigrams = Counter(words)

        # Create vocabulary
        self.vocabulary = set(self.unigrams)

        # Total number of words
        self.total_words = sum(self.unigrams.values())

    # --------------------------------------
    # Calculate P(correct word)
    # --------------------------------------

    def word_probability(self, word):

        # Add-one smoothing
        return (
            self.unigrams[word] + 1
        ) / (
            self.total_words
            + len(self.vocabulary)
            + 1
        )

    # --------------------------------------
    # Calculate P(error | correct word)
    # --------------------------------------

    @staticmethod
    def channel_probability(error, candidate):

        # Approximate uniform one-edit channel
        return 1.0 / (
            54 * max(len(candidate), 1)
        )

    # --------------------------------------
    # Find possible correct words
    # --------------------------------------

    def candidates(self, error):

        possible_words = edits1(error)

        # Only retain words present in vocabulary
        valid_candidates = (
            possible_words.intersection(self.vocabulary)
        )

        return valid_candidates

    # --------------------------------------
    # Select the most probable correction
    # --------------------------------------

    def correct(self, error):

        candidates = self.candidates(error)

        if not candidates:
            return error

        best_word = max(
            candidates,
            key=lambda candidate:

            self.word_probability(candidate)
            *
            self.channel_probability(error, candidate)
        )

        return best_word


# ==========================================
# 5. GENERATE SYNTHETIC SPELLING ERRORS
# ==========================================

def make_typo(word, rng):

    letters = string.ascii_lowercase

    operation = rng.choice([
        "delete",
        "insert",
        "substitute",
        "transpose"
    ])

    # Deletion
    if operation == "delete":

        i = rng.randrange(len(word))

        return word[:i] + word[i + 1:]

    # Insertion
    if operation == "insert":

        i = rng.randrange(len(word) + 1)

        ch = rng.choice(letters)

        return word[:i] + ch + word[i:]

    # Substitution
    if operation == "substitute":

        i = rng.randrange(len(word))

        choices = letters.replace(word[i], "")

        ch = rng.choice(choices)

        return word[:i] + ch + word[i + 1:]

    # Transposition
    if operation == "transpose":

        if len(word) < 2:
            return make_typo(word, rng)

        i = rng.randrange(len(word) - 1)

        return (
            word[:i]
            + word[i + 1]
            + word[i]
            + word[i + 2:]
        )


# ==========================================
# 6. EVALUATE SPELL CHECKER ACCURACY
# ==========================================

def evaluate(checker, test_size=TEST_SIZE):

    rng = random.Random(RANDOM_SEED)

    # Select words suitable for testing
    eligible = [
        word
        for word, count in checker.unigrams.items()
        if count >= 2 and len(word) >= 4
    ]

    examples = []

    # Generate spelling errors
    for _ in range(test_size):

        correct_word = rng.choice(eligible)

        misspelling = make_typo(correct_word, rng)

        examples.append(
            (misspelling, correct_word)
        )

    # Predict corrections
    correct_predictions = 0

    for error, actual_word in examples:

        predicted_word = checker.correct(error)

        if predicted_word == actual_word:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = (
        correct_predictions / len(examples)
    )

    print("\n========== EVALUATION RESULTS ==========")

    print("Total test examples:", len(examples))

    print("Correct predictions:", correct_predictions)

    print("Incorrect predictions:",
          len(examples) - correct_predictions)

    print("Accuracy: {:.2f}%".format(
        accuracy * 100
    ))

    return accuracy


# ==========================================
# 7. MAIN PROGRAM
# ==========================================

def main():

    pdf_path = PDF_FILE

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"Place {PDF_FILE!r} in the same folder "
            "as this Python script."
        )

    # Load dataset
    print("Loading Holbrook dataset...")

    words = extract_words(pdf_path)

    print("Dataset loaded successfully!")

    # Initialize spell checker
    checker = NoisyChannelSpellChecker(words)

    print("\n========== DATASET STATISTICS ==========")

    print("Total tokens:", len(words))

    print("Vocabulary size:",
          len(checker.vocabulary))

    # --------------------------------------
    # Test sample spelling errors
    # --------------------------------------

    print("\n========== SPELL CORRECTION ==========")

    test_words = [
        "rainning",
        "raning",
        "famly",
        "siter",
        "wakh"
    ]

    for error in test_words:

        correction = checker.correct(error)

        print(
            "{} --> {}".format(error, correction)
        )

    # --------------------------------------
    # Evaluate accuracy
    # --------------------------------------

    print("\n========== MODEL EVALUATION ==========")

    evaluate(checker)


# ==========================================
# 8. EXECUTE PROGRAM
# ==========================================

if __name__ == "__main__":

    main()
