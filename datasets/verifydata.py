import json

def verify_dataset(file_path):
    """
    Verify the content and structure of a JSONL dataset file.
    
    Args:
        file_path (str): Path to the JSONL file to verify
    """
    # Load the entire dataset file into memory
    # Each line is parsed as a JSON object and stored in a list
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = [json.loads(line) for line in file]

    # Iterate through each entry in the dataset to verify its structure
    for entry in dataset:
        # Check if the required keys ('InputText' and 'SentimentLabel') exist in each entry
        if 'InputText' not in entry or 'SentimentLabel' not in entry:
            # If any required key is missing, print the invalid entry and stop checking
            print(f"Missing keys in entry: {entry}")
            break
        else:
            # If the entry has all required keys, mark it as valid
            print(f"Entry is valid: {entry}")


if __name__ == "__main__":
    # Example usage of the verification function
    # The script expects a JSONL file named 'data.jsonl' in the same directory
    verify_dataset('data.jsonl')