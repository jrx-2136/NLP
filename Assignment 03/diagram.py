import importlib
import string

try:
    VisualDFA = importlib.import_module("visual_automata.fa.dfa").VisualDFA
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'visual-automata'. Install it with: pip install visual-automata"
    ) from exc

letters = set(string.ascii_lowercase)

transitions = {
    "q0": {letter: "q1" for letter in letters},
    "q1": {letter: "q1" for letter in letters}
}

dfa = VisualDFA(
    states={"q0", "q1"},
    input_symbols=letters,
    transitions=transitions,
    initial_state="q0",
    final_states={"q1"}
)

dfa.show_diagram(view=True)