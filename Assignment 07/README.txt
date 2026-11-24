HINDI POS TAGGING - LAB ASSIGNMENT 7

FILES
- naive_bayes.py : Question 1, Naive Bayes with features length, word, previous word, next word, previous tag.
- hmm_tagger.py  : Question 2, HMM with forward Viterbi, backward Viterbi, and forward-backward posterior decoding.
- pos_utils.py   : Shared dataset reader, reports, CSV export and plots.
- Hindi-POS-Data/ : training, dev and test files from the supplied ZIP.

SETUP
Install Python 3.9+ and packages:
    pip install scikit-learn matplotlib

RUN
From this directory:
    python naive_bayes.py
    python hmm_tagger.py

Outputs are written to results_nb/ and results_hmm/ respectively. Each evaluation prints accuracy and a per-tag precision/recall/F1 report, saves a per-tag F1 CSV, and saves charts.

DATA FORMAT
Each non-empty line must be: word<TAB>POS_TAG. Blank lines separate sentences.

NOTES
- Naive Bayes uses the gold previous tag to create training features and the predicted previous tag at inference time.
- Forward and backward Viterbi both find a globally best tag sequence using dynamic programming, with the recurrence traversed in opposite directions.
- Forward-backward uses posterior decoding (maximum marginal tag at each position); this is not the same as the globally best Viterbi sequence and may produce a different sequence.
- The model uses add-alpha smoothing (alpha=0.1) and an unknown-word token for emissions.
