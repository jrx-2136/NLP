import re
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / "brown_nouns.txt"

with DATA_FILE.open("r", encoding="utf-8") as file:
    words = {line.strip() for line in file if re.fullmatch(r"[a-z]+", line.strip())}

# --- FST Data Structures ---
# We use simple dictionaries instead of a complex Class to keep it beginner-friendly.
transitions = {}      # state -> { (input_char, output_char): next_state }
final_outputs = {}    # state -> set of grammatical tags (e.g., "+N+SG")
state_counter = 0

def new_state():
    global state_counter
    state = f"q{state_counter}"
    state_counter += 1
    transitions[state] = {}
    return state

start_state = new_state()

# --- 2. Build the Trie (Prefix Sharing) ---
# By checking if a transition already exists, we share prefixes automatically.
for word in words:
    
    # 2a. Add Singular Form
    curr = start_state
    for char in word:
        if (char, char) not in transitions[curr]:
            transitions[curr][(char, char)] = new_state()
        curr = transitions[curr][(char, char)]
    
    if curr not in final_outputs:
        final_outputs[curr] = set()
    final_outputs[curr].add("+N+SG")
    
    # 2b. Add Plural Form
    if word.endswith(("s", "x", "z", "ch", "sh")):
        plural = word + "es"
        plural_type = "es"
    elif word.endswith("y"):
        plural = word[:-1] + "ies"
        plural_type = "ies"
    else:
        plural = word + "s"
        plural_type = "s"
        
    curr = start_state
    for i, char in enumerate(plural):
        # We output the root letters and silence the plural suffixes using ε (epsilon)
        if plural_type == "ies" and i >= len(word) - 1:
            out_char = 'y' if char == 'i' else 'ε'
        elif plural_type == "es" and i >= len(word):
            out_char = 'ε'
        elif plural_type == "s" and i >= len(word):
            out_char = 'ε'
        else:
            out_char = char
            
        if (char, out_char) not in transitions[curr]:
            transitions[curr][(char, out_char)] = new_state()
        curr = transitions[curr][(char, out_char)]
        
    if curr not in final_outputs:
        final_outputs[curr] = set()
    final_outputs[curr].add("+N+PL")

states_before = len(transitions)

# --- 3. State Minimization (Merging Equivalent States) ---
signature_to_state = {}

def minimize(state):
    """
    Bottom-up recursive minimization. Two states are identical if their 
    outgoing transitions and final outputs are exactly the same.
    """
    # 1. Minimize all children first (bottom-up)
    for key, next_state in list(transitions[state].items()):
        transitions[state][key] = minimize(next_state)
        
    # 2. Build a unique "signature" for the current state
    trans_sig = tuple(sorted(transitions[state].items()))
    final_sig = tuple(sorted(final_outputs.get(state, [])))
    signature = (final_sig, trans_sig)
    
    # 3. If we've seen this exact signature before, merge! Return the old state.
    if signature in signature_to_state:
        return signature_to_state[signature]
    else:
        signature_to_state[signature] = state
        return state

# Run minimization from the start state
start_state = minimize(start_state)

# Count states after minimization by finding all reachable states
reachable = set()
def find_reachable(state):
    if state in reachable: return
    reachable.add(state)
    for next_state in transitions[state].values():
        find_reachable(next_state)

find_reachable(start_state)
states_after = len(reachable)

print(f"States BEFORE minimization: {states_before}")
print(f"States AFTER minimization:  {states_after}")
print(f"Total states reduced:       {states_before - states_after}")


# --- 4. Print Alphabets ---
input_alphabet = set()
output_alphabet = set()

for state in reachable:
    for (in_char, out_char) in transitions[state].keys():
        input_alphabet.add(in_char)
        output_alphabet.add(out_char)
        
    for tag in final_outputs.get(state, []):
        output_alphabet.add(tag)

print("\n--- Input Alphabet ---")
print(sorted(input_alphabet))

print("\n--- Output Alphabet ---")
print(sorted(output_alphabet))


# --- 5. Transition Table (Small Sample) ---
print("\n--- Transition Table (Sample) ---")
print(f"{'State':<10} | {'Input':<5} | {'Output':<10} | {'Next State / Final'}")
print("-" * 55)

count = 0
for state in reachable:
    for (in_char, out_char), next_state in transitions[state].items():
        if count < 15:
            print(f"{state:<10} | {in_char:<5} | {out_char:<10} | {next_state}")
            count += 1
            
    if state in final_outputs and count < 20:
        for tag in final_outputs[state]:
            print(f"{state:<10} | {'END':<5} | {tag:<10} | [FINAL]")
            count += 1
            
    if count >= 20:
        break


# --- 6. Analyze Function ---
def analyze(word):
    # Queue stores: (current_state, current_index_in_word, accumulated_output)
    queue = [(start_state, 0, "")]
    results = set()
    
    while queue:
        state, idx, out_str = queue.pop(0)
        
        # Reached the end of the input word
        if idx == len(word):
            if state in final_outputs:
                for tag in final_outputs[state]:
                    results.add(out_str + tag)
            continue
            
        char = word[idx]
        for (in_char, out_char), next_state in transitions.get(state, {}).items():
            if in_char == char:
                # If output is epsilon (ε), we don't add it to the final string
                actual_out = "" if out_char == "ε" else out_char
                queue.append((next_state, idx + 1, out_str + actual_out))
                
    return ", ".join(sorted(results)) if results else "Invalid Word"


test_words = [
    "fox", "foxes", "bag", "bags", "try", "tries", 
    "dish", "dishes", "foxs", "dishs","crannies","writers"
]

print("\n--- Test Results ---")
for word in test_words:
    print(f"{word:<8} -> {analyze(word)}")