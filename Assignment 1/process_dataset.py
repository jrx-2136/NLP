from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from tokenizer import sentence_tokenizer, word_tokenizer


def _paragraphs_from_text(input_path):
    paragraph = []
    with input_path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                paragraph.append(line)
            elif paragraph:
                yield " ".join(paragraph)
                paragraph = []

    if paragraph:
        yield " ".join(paragraph)


def process_dataset(input_file, output_parquet, target_sentences=1_000_000, batch_size=10_000):
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path

    output_path = Path(output_parquet)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_sentences = 0
    total_words = 0
    total_characters = 0
    unique_tokens = set()
    tokenized_batch = []
    writer = None

    print(f"Processing up to {target_sentences:,} sentences from {input_path}...")
    progress = tqdm(total=target_sentences, unit="sentences")

    try:
        for paragraph in _paragraphs_from_text(input_path):
            for sentence in sentence_tokenizer(paragraph):
                tokens = word_tokenizer(sentence)
                if not tokens:
                    continue

                tokenized_batch.append(" ".join(tokens))
                total_sentences += 1
                total_words += len(tokens)
                total_characters += sum(len(token) for token in tokens)
                unique_tokens.update(tokens)
                progress.update(1)

                if len(tokenized_batch) >= batch_size:
                    table = pa.Table.from_pandas(
                        pd.DataFrame({"tokenized_sentence": tokenized_batch}),
                        preserve_index=False,
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                    writer.write_table(table)
                    tokenized_batch.clear()

                if total_sentences >= target_sentences:
                    break

            if total_sentences >= target_sentences:
                break
    finally:
        if tokenized_batch:
            table = pa.Table.from_pandas(
                pd.DataFrame({"tokenized_sentence": tokenized_batch}),
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
        if writer is not None:
            writer.close()
        progress.close()

    if total_sentences < target_sentences:
        raise ValueError(
            f"Input contains only {total_sentences:,} tokenized sentences; "
            f"{target_sentences:,} were requested."
        )

    stats = {
        "Total Sentences": total_sentences,
        "Total Words": total_words,
        "Total Characters (in words)": total_characters,
        "Average Sentence Length": total_words / total_sentences,
        "Average Word Length": total_characters / total_words if total_words else 0,
        "Type/Token Ratio (TTR)": len(unique_tokens) / total_words if total_words else 0,
    }

    print("\n--- Corpus Statistics ---")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\nSaved {total_sentences:,} sentences to {output_path}")
    return stats