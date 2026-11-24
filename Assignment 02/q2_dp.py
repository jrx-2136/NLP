import json
import math

# Load dataset
with open("Assignment 2\\text_segmentation_dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

word_counts = data["word_counts"]
#print(word_counts)
test_cases = data["test_cases"]
#print(test_cases)


total_words = sum(word_counts.values())

# Find longest word
max_word_length = max(len(word) for word in word_counts)


# DYNAMIC PROGRAMMING SEGMENTATION

def dp_segmentation(text):

    n = len(text)

    # dp[i] = best score for first i characters
    dp = [float("-inf")] * (n + 1)

    # Store previous position and word
    previous = [None] * (n + 1)

    # Empty text has score 0
    dp[0] = 0

    # Go through every position
    for i in range(1, n + 1):

        # Try different word lengths
        max_length = min(max_word_length, i)

        for length in range(1, max_length + 1):

            word = text[i - length:i]

            # Check if word exists
            if word in word_counts:

                # Calculate probability
                probability = word_counts[word] / total_words

                # Calculate log probability
                log_probability = math.log(probability)

                # Calculate total score
                score = dp[i - length] + log_probability

                # Keep best score
                if score > dp[i]:

                    dp[i] = score

                    previous[i] = (
                        i - length,
                        word
                    )

        # If no word was found
        if previous[i] is None:

            dp[i] = dp[i - 1] - 100

            previous[i] = (
                i - 1,
                text[i - 1]
            )


    # BACKTRACKING


    words = []

    i = n

    while i > 0:

        previous_position, word = previous[i]

        words.append(word)

        i = previous_position

    # Reverse the result
    words.reverse()

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


correct = 0
total_edit_distance = 0

# File to store detailed intermediate output
output_file = "dp_output.txt"

# Store results for JSON
results = []

with open(output_file, "w", encoding="utf-8") as file:


    file.write("DYNAMIC PROGRAMMING - INTERMEDIATE OUTPUT\n")


    for index, test in enumerate(test_cases, start=1):

        text = test["input"]



        predicted = dp_segmentation(text)

        file.write("\n-----------------------------------\n")
        file.write(f"Test Case: {index}\n")
        file.write("-----------------------------------\n")

        file.write("\nInput:\n")
        file.write(text)

        file.write("\n\nDP Prediction:\n")
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

        # STEP 4: EDIT DISTANCE
     

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


    # RESULTS


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


final_results = {

    "method": "Dynamic Programming",

    "total_test_cases": total,

    "correct_cases": correct,

    "accuracy": round(accuracy, 2),

    "average_edit_distance": round(
        average_edit_distance, 3
    ),

    "test_cases": results
}


with open(
    "dp_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_results,
        file,
        indent=4,
        ensure_ascii=False
    )

# DISPLAY FINAL RESULTS ON SCREEN


print("DYNAMIC PROGRAMMING")

print("Total Test Cases:", total)

print("Correct Cases:", correct)

print("Accuracy:", round(accuracy, 2), "%")

print(
    "Average Edit Distance:",
    round(average_edit_distance, 3)
)

#print("\nDetailed output saved to: dp_output.txt")
#print("Results saved to: dp_results.json")