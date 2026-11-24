# Assignment 3: Finite-State Methods

This assignment contains finite-state exercises for recognizing lowercase words and constructing an English noun morphological transducer.

## Files

- `q1.py` — interactive deterministic finite automaton; accepts a string containing one or more lowercase letters.
- `q2.py` — builds a finite-state transducer for singular and plural noun forms, performs state minimization, and analyzes test words.
- `diagram.py` — visualizes the accepted-language DFA using `visual-automata`.
- `brown_nouns.txt` — noun vocabulary used to build the transducer.
- `tempCodeRunnerFile.py` — temporary editor-runner scratch file.

## Behavior

`q1.py` models a state that enters on the first lowercase letter and remains accepting while subsequent characters are lowercase. `q2.py` creates trie transitions for nouns, generates regular plural forms, uses epsilon output for part of plural suffixes, and merges equivalent states. It reports state counts, alphabets, a transition sample, and singular/plural analyses.

## Run

```bash
python "Assignment 3/q1.py"
python "Assignment 3/q2.py"
pip install visual-automata
python "Assignment 3/diagram.py"
```
