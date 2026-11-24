import json

# Load dataset
with open("Assignment 2\\text_segmentation_dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

word_counts = data["word_counts"]
test_cases = data["test_cases"]

vocabulary = set(word_counts.keys())

# Find longest word in vocabulary
max_word_length = max(len(word) for word in vocabulary)


# GREEDY SEGMENTATION

def greedy_segmentation(text):

    words = []
    i = 0

    while i < len(text):

        found_word = None

        # Start from longest possible word
        max_length = min(max_word_length, len(text) - i)

        for length in range(max_length, 0, -1):

            word = text[i:i + length]

            if word in vocabulary:
                found_word = word
                break

        # If no word is found
        if found_word is None:
            found_word = text[i]

        words.append(found_word)

        i += len(found_word)

    return words


# EDIT DISTANCE

def edit_distance(actual, predicted):

    m = len(actual)
    n = len(predicted)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if actual[i - 1] == predicted[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n]

# INTERMEDIATE OUTPUT + EVALUATION


correct = 0
total_edit_distance = 0

# File to store detailed intermediate output
output_file = "greedy_output.txt"

# Store results for JSON
results = []

with open(output_file, "w", encoding="utf-8") as file:

    file.write("===================================\n")
    file.write("GREEDY LONGEST MATCH - INTERMEDIATE OUTPUT\n")
    file.write("===================================\n")

    for index, test in enumerate(test_cases, start=1):

        text = test["input"]

        # ----------------------------------------------------
        # STEP 1: GREEDY ALGORITHM PREDICTION
        # ----------------------------------------------------

        predicted = greedy_segmentation(text)

        file.write("\n-----------------------------------\n")
        file.write(f"Test Case: {index}\n")
        file.write("-----------------------------------\n")

        file.write("\nInput:\n")
        file.write(text)

        file.write("\n\nGreedy Prediction:\n")
        file.write(str(predicted))



        actual = test["ground_truth"].split()

        file.write("\n\nGround Truth:\n")
        file.write(str(actual))


        if predicted == actual:

            verification = "CORRECT"
            correct += 1

        else:

            verification = "INCORRECT"

        file.write("\n\nVerification: ")
        file.write(verification)

        # ----------------------------------------------------
        # STEP 4: EDIT DISTANCE
        # ----------------------------------------------------

        distance = edit_distance(
            actual,
            predicted
        )

        file.write(f"\nEdit Distance: {distance}\n")

        total_edit_distance += distance

        # Store individual test case result
        results.append({
            "test_case": index,
            "input": text,
            "prediction": predicted,
            "ground_truth": actual,
            "verification": verification,
            "edit_distance": distance
        })

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    total = len(test_cases)

    accuracy = (correct / total) * 100

    average_edit_distance = total_edit_distance / total

    file.write("\n\n===================================\n")
    file.write("FINAL RESULTS\n")
    file.write("===================================\n")

    file.write(f"Total Test Cases: {total}\n")
    file.write(f"Correct Cases: {correct}\n")
    file.write(f"Accuracy: {round(accuracy, 2)} %\n")
    file.write(
        f"Average Edit Distance: {round(average_edit_distance, 3)}\n"
    )


# ------------------------------------------------------------
# SAVE FINAL RESULTS AS JSON
# ------------------------------------------------------------

final_results = {

    "method": "Greedy Longest Match",

    "total_test_cases": total,

    "correct_cases": correct,

    "accuracy": round(accuracy, 2),

    "average_edit_distance": round(
        average_edit_distance, 3
    ),

    "test_cases": results
}


with open(
    "greedy_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_results,
        file,
        indent=4,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# DISPLAY FINAL RESULTS ON SCREEN
# ------------------------------------------------------------

print("\n===================================")
print("GREEDY LONGEST MATCH")
print("===================================")

print("Total Test Cases:", total)

print("Correct Cases:", correct)

print("Accuracy:", round(accuracy, 2), "%")

print(
    "Average Edit Distance:",
    round(average_edit_distance, 3)
)

#print("\nDetailed output saved to: greedy_output.txt")
#print("Results saved to: greedy_results.json")