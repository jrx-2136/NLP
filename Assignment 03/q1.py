TRANSITIONS = {
    "q0": {"a-z": "q1"},
    "q1": {"a-z": "q1"},
}

def print_transition_table():
    print("\nTransition Table\n")
    print(f"{'Current State':<15} | {'Input':<10} | {'Next State':<10}")
    print("-" * 44)
    for state, rules in TRANSITIONS.items():
        for in_symbol, next_state in rules.items():
            print(f"{state:<15} | {in_symbol:<10} | {next_state:<10}")


def is_valid_word(word):
    state = "q0"
    for char in word:
        if state == "q0":
            if 'a' <= char <= 'z':
                state = "q1"
            else:
                return False

        elif state == "q1":
            if 'a' <= char <= 'z':
                state = "q1"
            else:
                return False

    return state == "q1"


print_transition_table()
print()
word = input("word: ")

if is_valid_word(word):
    print("accepted")
else:
    print("not accepted")