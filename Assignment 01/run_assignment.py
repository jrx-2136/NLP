import json
from pathlib import Path

from process_dataset import process_dataset


def main():
    assignment_dir = Path(__file__).resolve().parent
    input_file = assignment_dir.parent / "mr.txt"
    output_file = assignment_dir / "indiccorp_tokenized.parquet"
    stats_file = assignment_dir / "stats.json"

    print("=" * 80)
    print("Running Text Preprocessing Pipeline")
    print("=" * 80)

    stats = process_dataset(
        input_file=input_file,
        output_parquet=output_file,
        target_sentences=1_000_000,
    )

    with stats_file.open("w", encoding="utf-8") as output:
        json.dump(stats, output, indent=4, ensure_ascii=False)

    print(f"\nPipeline complete. Statistics saved to {stats_file}")


if __name__ == "__main__":
    main()