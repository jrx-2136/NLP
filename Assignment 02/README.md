# Assignment 2: Word Segmentation

This assignment compares two algorithms that recover word boundaries in an unsegmented text string.

## Files

- `text_segmentation_dataset.json` — vocabulary frequencies and test cases containing input strings and ground-truth segmentation.
- `q1_greedy.py` — longest-match-first greedy segmentation.
- `q2_dp.py` — dynamic programming segmentation using word log probabilities.
- `greedy_output.txt` and `dp_output.txt` — detailed per-test and aggregate results.
- `greedy_results.json` and `dp_results.json` — structured accuracy and edit-distance summaries.
- `download_data.py` — dataset acquisition helper, if needed.

## Methods

The greedy method repeatedly selects the longest vocabulary word that matches at the current position. The dynamic-programming method evaluates every valid word split and retains the maximum combined log probability. Both implementations calculate exact edit distance between predicted and ground-truth token sequences.

## Run

From the repository root:

```bash
python "Assignment 2/q1_greedy.py"
python "Assignment 2/q2_dp.py"
```

The scripts print final accuracy and average edit distance, then write the text and JSON reports.
