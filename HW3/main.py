def create_vocab(train_file, output_file, threshold=3):
    """
    Create a vocabulary from the training data:
    1. Count the occurrences of each word.
    2. Replace low-frequency words (occurrences below the threshold) with <unk>.
    3. The generated vocabulary format:
       - The first line contains <unk> and its total occurrence count.
       - Other words are sorted in descending order of frequency.
       - Each line follows the format: word \t index \t count.
    """
    word_counts = {}

    # Read training data and count word frequencies
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # Empty line indicates end of a sentence
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue  # Skip lines that do not match the expected format
            _, word, _ = parts
            word_counts[word] = word_counts.get(word, 0) + 1

    # Construct vocabulary, grouping low-frequency words as <unk>
    unk_count = 0
    vocab = {}
    for word, count in word_counts.items():
        if count < threshold:
            unk_count += count
        else:
            vocab[word] = count

    # Create final vocabulary list: first entry is <unk>, followed by words sorted by frequency
    final_vocab = []
    final_vocab.append(("<unk>", unk_count))
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    final_vocab.extend(sorted_vocab)

    # Write to output file vocab.txt, each line in the format: word \t index \t count
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, (word, count) in enumerate(final_vocab):
            out_f.write(f"{word}\t{idx}\t{count}\n")

    print(f"Vocabulary generated successfully. Total vocabulary size: {len(final_vocab)}, <unk> occurrences: {unk_count}")

train_file = "./data/train"  # Path to training data file
output_file = "vocab.txt"    # Path to output vocabulary file
threshold = 3                # Words with occurrences below this threshold are replaced with <unk>
create_vocab(train_file, output_file, threshold)


import json

def learn_hmm(train_file, output_file):
    """
    Learn an HMM model from the training data, computing transition and emission probabilities,
    and save the model as a JSON file.

    Transition Probability: t(s′|s) = count(s→s′) / (sum of count(s→*))
    Emission Probability: e(x|s) = count(s→x) / (count(s) for emission)
    """
    # Dictionary to count emissions: key is (tag, word)
    emission_counts = {}
    # Dictionary to store total occurrences of each tag (for emission probability denominator)
    tag_total_emission = {}

    # Dictionary to count transitions: key is (prev_tag, curr_tag)
    transition_counts = {}
    # Dictionary to store total transitions from each tag (for transition probability denominator)
    transition_total = {}

    # Read data sentence by sentence (sentences are separated by empty lines)
    with open(train_file, 'r', encoding='utf-8') as f:
        sentence_tags = []
        for line in f:
            line = line.strip()
            if line == "":
                # Compute transitions for the current sentence
                for i in range(len(sentence_tags) - 1):
                    prev_tag = sentence_tags[i]
                    curr_tag = sentence_tags[i + 1]
                    transition_counts[(prev_tag, curr_tag)] = transition_counts.get((prev_tag, curr_tag), 0) + 1
                    transition_total[prev_tag] = transition_total.get(prev_tag, 0) + 1
                sentence_tags = []
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                continue  # Skip lines that do not match the expected format
            _, word, tag = parts

            # Update emission counts
            emission_counts[(tag, word)] = emission_counts.get((tag, word), 0) + 1
            tag_total_emission[tag] = tag_total_emission.get(tag, 0) + 1

            # Store tag sequence for transition counting
            sentence_tags.append(tag)

        # Process the last sentence (if there is no empty line at the end of the file)
        if sentence_tags:
            for i in range(len(sentence_tags) - 1):
                prev_tag = sentence_tags[i]
                curr_tag = sentence_tags[i + 1]
                transition_counts[(prev_tag, curr_tag)] = transition_counts.get((prev_tag, curr_tag), 0) + 1
                transition_total[prev_tag] = transition_total.get(prev_tag, 0) + 1

    # Compute transition probabilities
    transition_prob = {}
    for (prev_tag, curr_tag), count in transition_counts.items():
        total = transition_total[prev_tag]
        # Format the key as "(prev_tag, curr_tag)"
        key = f"({prev_tag}, {curr_tag})"
        transition_prob[key] = count / total

    # Compute emission probabilities
    emission_prob = {}
    for (tag, word), count in emission_counts.items():
        total = tag_total_emission[tag]
        key = f"({tag}, {word})"
        emission_prob[key] = count / total

    # Construct the HMM model as a JSON object
    model = {
        "transition": transition_prob,
        "emission": emission_prob
    }

    # Save the model to a JSON file
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(model, out_f, indent=4)

    print(f"HMM model generated successfully. Transition parameters: {len(transition_prob)}, Emission parameters: {len(emission_prob)}")

train_file = "./data/train"  # Path to the training data file
output_file = "hmm.json"     # Path to the output model file
learn_hmm(train_file, output_file)

import subprocess

def evaluate_predictions(gold_file, pred_file):
    """
    Call eval.py to evaluate the prediction file against the gold standard.

    Parameters:
      gold_file (str): Path to the gold standard file.
      pred_file (str): Path to the prediction file (viterbi.out or greedy.out).
    """
    # Run the eval.py script with the given arguments
    result = subprocess.run(
        ["python", "eval.py", "-g", gold_file, "-p", pred_file],
        capture_output=True,
        text=True
    )
    # Print the evaluation results
    print(f"Evaluation results for {pred_file}:")
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

import json

def greedy_decode(test_file, output_file, model_file):
    """
    Perform part-of-speech tagging on test data using the greedy decoding algorithm 
    and save the predictions to the output file.

    Parameters:
    - test_file: Path to the test data file (contains index and word, sentences separated by empty lines)
    - output_file: Path to the output file, following the same format as training data 
                  (each line: "index\tword\tpredicted_tag")
    - model_file: JSON file containing the HMM model (includes transition and emission probabilities)
    """
    # Load the HMM model (transition and emission probabilities stored as string keys like "(NN, dog)")
    with open(model_file, 'r', encoding='utf-8') as f:
        model = json.load(f)
    transition = model["transition"]
    emission = model["emission"]

    # Convert string keys to tuples for easier lookup
    def parse_key(key):
        # Key format: "(tag, word)", remove parentheses and split by ", "
        key = key.strip("()")
        parts = key.split(", ")
        return tuple(parts)
    
    transition_prob = {parse_key(key): prob for key, prob in transition.items()}
    emission_prob = {parse_key(key): prob for key, prob in emission.items()}

    # Extract possible tags from emission probabilities
    possible_tags = list({tag for (tag, _) in emission_prob.keys()})

    # Function to get emission probability
    def get_emission(tag, word):
        # Try (tag, word), otherwise use (tag, "<unk>")
        if (tag, word) in emission_prob:
            return emission_prob[(tag, word)]
        elif (tag, "<unk>") in emission_prob:
            return emission_prob[(tag, "<unk>")]
        else:
            return 1e-10  # Very small probability for unseen words

    # Function to get transition probability
    def get_transition(prev_tag, tag):
        return transition_prob.get((prev_tag, tag), 1e-10)

    # Read test data and perform greedy decoding
    output_lines = []
    with open(test_file, 'r', encoding='utf-8') as f:
        sentence = []  # Store the current sentence as (index, word) tuples
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                # End of current sentence, perform prediction
                if sentence:
                    predicted_tags = []
                    prev_tag = None
                    for idx, (index, word) in enumerate(sentence):
                        best_tag = None
                        best_score = -1
                        for tag in possible_tags:
                            if idx == 0:
                                # First word: consider only emission probability
                                score = get_emission(tag, word)
                            else:
                                score = get_transition(prev_tag, tag) * get_emission(tag, word)
                            if score > best_score:
                                best_score = score
                                best_tag = tag
                        predicted_tags.append(best_tag)
                        prev_tag = best_tag
                    # Write predictions to output list (same format as training data)
                    for (index, word), tag in zip(sentence, predicted_tags):
                        output_lines.append(f"{index}\t{word}\t{tag}")
                    output_lines.append("")  # Empty line separates sentences
                    sentence = []
            else:
                # Test data format: each line contains index and word (ignore third column if present)
                parts = line.split('\t')
                if len(parts) >= 2:
                    index = parts[0]
                    word = parts[1]
                    sentence.append((index, word))

        # Process the last sentence if the file does not end with an empty line
        if sentence:
            predicted_tags = []
            prev_tag = None
            for idx, (index, word) in enumerate(sentence):
                best_tag = None
                best_score = -1
                for tag in possible_tags:
                    if idx == 0:
                        score = get_emission(tag, word)
                    else:
                        score = get_transition(prev_tag, tag) * get_emission(tag, word)
                    if score > best_score:
                        best_score = score
                        best_tag = tag
                predicted_tags.append(best_tag)
                prev_tag = best_tag
            for (index, word), tag in zip(sentence, predicted_tags):
                output_lines.append(f"{index}\t{word}\t{tag}")
            output_lines.append("")

    # Save predictions to the output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(output_lines))

    print(f"Greedy decoding completed. Predictions saved to {output_file}")

# Update file paths as needed:
test_file = "./data/dev"     # Test data file (each line: index \t word, sentences separated by empty lines)
output_file = "dev_greedy.out"    # Output file for predictions
model_file = "hmm.json"       # Previously generated HMM model file
greedy_decode(test_file, output_file, model_file)
evaluate_predictions("./data/dev", output_file)

# Test File
test_file = "./data/test"     # Test data file (each line: index \t word, sentences separated by empty lines)
output_file = "greedy.out"    # Output file for predictions
model_file = "hmm.json"       # Previously generated HMM model file
greedy_decode(test_file, output_file, model_file)


import json

def viterbi_decode(test_file, output_file, model_file):
    """
    Perform part-of-speech tagging on test data using the Viterbi decoding algorithm
    and save the results to the output file.

    Parameters:
      - test_file: Path to the test data file (each line format: index \t word, sentences separated by empty lines)
      - output_file: Path to the output file, following the same format as the training data 
                     ("index \t word \t predicted_tag")
      - model_file: JSON file containing the HMM model (transition and emission probabilities)
    """
    # Load the HMM model
    with open(model_file, 'r', encoding='utf-8') as f:
        model = json.load(f)
    transition = model["transition"]
    emission = model["emission"]

    # Convert string keys "(tag, word)" into tuples (tag, word) for easy lookup
    def parse_key(key):
        key = key.strip("()")
        parts = key.split(", ")
        return tuple(parts)

    transition_prob = {parse_key(key): prob for key, prob in transition.items()}
    emission_prob = {parse_key(key): prob for key, prob in emission.items()}

    # Extract all possible tags from emission probabilities
    possible_tags = list({tag for (tag, _) in emission_prob.keys()})

    # Function to get emission probability
    def get_emission(tag, word):
        # Try (tag, word) first, if not found, try (tag, "<unk>"), otherwise return a very small probability
        return emission_prob.get((tag, word), emission_prob.get((tag, "<unk>"), 1e-10))

    # Function to get transition probability
    def get_transition(prev_tag, tag):
        return transition_prob.get((prev_tag, tag), 1e-10)

    # Viterbi algorithm for decoding, input is a sentence (list of (index, word) tuples)
    def viterbi(sentence):
        n = len(sentence)
        # dp[t][tag] stores the highest probability of reaching tag at time step t
        dp = [{} for _ in range(n)]
        # backpointer[t][tag] stores the best previous tag leading to the current tag
        backpointer = [{} for _ in range(n)]
        
        # Initialization: First word only considers emission probability
        first_word = sentence[0][1]
        for tag in possible_tags:
            dp[0][tag] = get_emission(tag, first_word)
            backpointer[0][tag] = None
        
        # Dynamic programming step
        for t in range(1, n):
            word = sentence[t][1]
            for curr_tag in possible_tags:
                best_score = -1
                best_prev = None
                for prev_tag in possible_tags:
                    score = dp[t-1][prev_tag] * get_transition(prev_tag, curr_tag) * get_emission(curr_tag, word)
                    if score > best_score:
                        best_score = score
                        best_prev = prev_tag
                dp[t][curr_tag] = best_score
                backpointer[t][curr_tag] = best_prev

        # Termination: Find the best tag at the last time step
        best_last_tag = max(dp[n-1], key=dp[n-1].get)
        
        # Backtrack to get the best tag sequence
        best_tags = [None] * n
        best_tags[n-1] = best_last_tag
        for t in range(n-1, 0, -1):
            best_tags[t-1] = backpointer[t][best_tags[t]]
        return best_tags

    # Read test data, decode each sentence, and save the predictions
    output_lines = []
    with open(test_file, 'r', encoding='utf-8') as f:
        sentence = []  # Store the current sentence as (index, word) tuples
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if sentence:
                    predicted_tags = viterbi(sentence)
                    for (index, word), tag in zip(sentence, predicted_tags):
                        output_lines.append(f"{index}\t{word}\t{tag}")
                    output_lines.append("")  # Empty line separates sentences
                    sentence = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    index = parts[0]
                    word = parts[1]
                    sentence.append((index, word))

        # Process the last sentence if the file does not end with an empty line
        if sentence:
            predicted_tags = viterbi(sentence)
            for (index, word), tag in zip(sentence, predicted_tags):
                output_lines.append(f"{index}\t{word}\t{tag}")
            output_lines.append("")

    # Write predictions to the output file
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("\n".join(output_lines))

    print(f"Viterbi decoding completed. Predictions saved to {output_file}")

# Update file paths as needed
test_file = "./data/dev"     # Test data file, each line in the format: "index \t word", sentences separated by empty lines
output_file = "dev_viterbi.out"   # Output file for predictions
model_file = "hmm.json"       # HMM model file
viterbi_decode(test_file, output_file, model_file)
evaluate_predictions("./data/dev", output_file)

# Test file
test_file = "./data/test"     # Test data file, each line in the format: "index \t word", sentences separated by empty lines
output_file = "viterbi.out"   # Output file for predictions
model_file = "hmm.json"       # HMM model file
viterbi_decode(test_file, output_file, model_file)
