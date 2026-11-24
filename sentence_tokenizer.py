import re

# Common Hindi/English abbreviations
ABBREVIATIONS = [
    "श्री.",
    "डॉ.",
    "डा.",
    "प्रो.",
    "सुश्री.",
    "चौ.",
    "कु.",
    "वि.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr."
]

PLACEHOLDER = "<DOT>"

def sentence_tokenizer(paragraph):

    if paragraph is None:
        return []

    paragraph = str(paragraph).strip()

    if paragraph == "":
        return []

    # Protect abbreviations
    for abbr in ABBREVIATIONS:
        paragraph = paragraph.replace(abbr, abbr.replace(".", PLACEHOLDER))

    # Protect decimal numbers
    paragraph = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', paragraph)

    # Split into sentences
    sentences = re.split(r'(?<=[।!?.])\s+', paragraph)

    result = []

    for sentence in sentences:

        sentence = sentence.replace(PLACEHOLDER, ".")
        sentence = sentence.replace("<DECIMAL>", ".")

        sentence = sentence.strip()

        if sentence:
            result.append(sentence)

    return result