# Assignment 1: Text Preprocessing

This folder contains the codes and results for the first NLP assignment (Text Preprocessing).

## Files

- `tokenizer.py`: Contains a custom `sentence_tokenizer` (splits paragraphs into sentences) and `word_tokenizer` (uses Regex to extract URLs, emails, dates, decimals, words, and punctuations).
- `process_dataset.py`: Streams paragraphs from `../mr.txt`, processes exactly 1,000,000 sentences with the tokenizers, computes the requested corpus statistics, and writes the tokenized output to a `.parquet` file.
- `run_assignment.py`: Main execution script for the Marathi corpus.
- `stats.json`: Contains the final computed statistics in JSON format.
- `indiccorp_tokenized.parquet`: The processed Marathi data from `mr.txt`, where each row contains one tokenized sentence (tokens separated by spaces).

## Instructions to Run

1. Make sure you have the required dependencies:
   ```bash
   pip install pandas pyarrow tqdm
   ```
2. Run the main pipeline script:
   ```bash
   python run_assignment.py
   ```
   The script reads `../mr.txt` as UTF-8, stops after 1,000,000 tokenized sentences, writes `indiccorp_tokenized.parquet`, and records statistics in `stats.json`.

## Corpus Statistics Results

### Marathi corpus

- **Total number of sentences:** 1,000,000
- **Total number of words (tokens):** 13,787,547
- **Total number of characters:** 70,817,359
- **Average Sentence Length:** 13.7875 words per sentence
- **Average Word Length:** 5.1363 characters per word
- **Type/Token Ratio (TTR):** 0.0467

The OSCAR-2301 repeat (Question 2) is intentionally not included.
