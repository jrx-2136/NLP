# NLP Lab Assignment 9 — Solution

## Dataset
The assignment ZIP contains `Hindi-POS-Data.zip`, with:
- `hindi-train-pos.txt`
- `hindi-dev-pos.txt`
- `hindi-test-pos.txt`

Extract these three files into a folder named `Hindi-POS-Data`.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run
```bash
python assignment9_solution.py --data_dir ./Hindi-POS-Data --output_dir ./assignment9_results
```

If training time or RAM is limited:
```bash
python assignment9_solution.py --skip_crf
```
or
```bash
python assignment9_solution.py --skip_memm
```

## Output
The results directory contains:
- overall_comparison.csv
- all_models_per_tag_comparison.csv
- per-model dev/test per-tag CSV files
- comparison_accuracy.png
- comparison_macro_f1.png
- comparison_weighted_f1.png
- per_tag_f1_dev.png
- per_tag_f1_test.png
- saved model files (`*.joblib`)
- experiment_summary.json

## Important implementation note
The linked `nltk-maxent-pos-tagger` repository is an older Python-2-era NLTK implementation and configures an external MEGAM executable. The MEMM in this solution preserves the maximum-entropy Markov formulation using multinomial logistic regression, with the previous tag as a conditioning feature. Training uses gold previous tags; inference uses previously predicted tags. This makes it executable in current Python environments.

The CRF follows the sklearn-crfsuite sequence-labeling interface and uses the assignment's exact word-context, prefix, suffix, and length features.

Naive Bayes and HMM are implemented in this same script so all four models use the supplied Hindi splits and are evaluated consistently. If your instructor requires the exact Assignment 7 implementations rather than reproducible baselines, replace these two classes with your Assignment 7 code while keeping the evaluation functions.
