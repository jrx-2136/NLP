from datasets import load_dataset
from tqdm import tqdm
import os

# ==============================
# Configuration
# ==============================

LANGUAGE = "mar_Deva"          # Change if needed
OUTPUT_FILE = "../data/raw.txt"
TARGET_LINES = 100000

# ==============================
# Count existing lines
# ==============================

existing = 0

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        existing = sum(1 for _ in f)

print(f"Already saved: {existing} lines")

if existing >= TARGET_LINES:
    print("Target already reached.")
    exit()

# ==============================
# Stream Dataset
# ==============================


dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    "indiccorp_v2",
    data_dir="data/hin_Deva",
    split="train",
    streaming=True,
)

# ==============================
# Save Data
# ==============================

progress = tqdm(total=TARGET_LINES)

progress.update(existing)

count = existing

with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:

    skipped = 0

    for sample in dataset:

        if skipped < existing:
            skipped += 1
            continue

        text = sample["text"].strip()

        if text:

            outfile.write(text + "\n")

            count += 1

            progress.update(1)

        if count >= TARGET_LINES:
            break

progress.close()

print()
print("Finished!")
print("Saved:", count, "paragraphs")