import re

# Common abbreviations to prevent them from splitting sentences
ABBREVIATIONS = [
    "श्री.", "डॉ.", "डा.", "प्रो.", "सुश्री.", "चौ.", "कु.", "वि.",
    "Mr.", "Mrs.", "Ms.", "Dr."
]

ABBREVIATION_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(abbr) for abbr in ABBREVIATIONS) + r')'
)

DECIMAL_PATTERN = re.compile(r'(\d)\.(\d)')

TOKEN_PATTERN = re.compile(
    r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    r'|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    r'|\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'
    r'|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b'
    r'|[.,!?;:/()\[\]{}"\'…।॥\-]'
    r'|[^\s.,!?;:/()\[\]{}"\'…।॥\-]+'
)

URL_TRAILING_PUNCTUATION = ".,!?;:)]}"


def _protect_abbreviations(paragraph):
    return ABBREVIATION_PATTERN.sub(lambda match: match.group(0).replace('.', '<DOT>'), paragraph)

def sentence_tokenizer(paragraph):
    """
    Tokenizes a paragraph into sentences.
    """
    if not paragraph:
        return []

    paragraph = str(paragraph).strip()
    if not paragraph:
        return []

    # Protect abbreviations and decimals before splitting on sentence boundaries.
    paragraph = _protect_abbreviations(paragraph)

    # Protect decimal numbers (e.g. 1.5)
    paragraph = DECIMAL_PATTERN.sub(r'\1<DECIMAL>\2', paragraph)

    # Split into sentences based on Hindi and English sentence terminators
    sentences = re.split(r'(?<=[।!?.])\s+', paragraph)

    result = []
    for sentence in sentences:
        # Restore placeholders
        sentence = sentence.replace("<DOT>", ".")
        sentence = sentence.replace("<DECIMAL>", ".")
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)

    return result

def word_tokenizer(sentence):
    """
    Tokenizes a sentence into words, numbers, decimals, emails, urls, dates, and punctuations.
    """
    if not sentence:
        return []

    tokens = []
    for match in TOKEN_PATTERN.finditer(sentence):
        token = match.group(0)
        if token.strip():
            if token.startswith(("http://", "https://", "www.")):
                trailing = ""
                while token and token[-1] in URL_TRAILING_PUNCTUATION:
                    trailing = token[-1] + trailing
                    token = token[:-1]
                if token:
                    tokens.append(token)
                tokens.extend(trailing)
                continue
            tokens.append(token)
            
    return tokens
