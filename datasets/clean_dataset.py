import json
from typing import List, Dict

def clean_dataset(input_file: str, output_file: str) -> None:
    """
    Clean a dataset by validating and processing JSON records from an input file
    and saving the cleaned data to an output file.

    Args:
        input_file (str): Path to the input JSONL file containing the dataset
        output_file (str): Path where the cleaned dataset will be saved

    The function expects each line in the input file to be a valid JSON object
    containing 'InputText' and 'SentimentLabel' fields.
    """
    cleaned_data: List[Dict] = []

    try:
        # Load and process the dataset
        dataset = load_dataset(input_file)
        
        # Clean and validate each entry
        cleaned_data = process_entries(dataset)

        # Save the cleaned dataset
        save_dataset(cleaned_data, output_file)

        print(f"Successfully cleaned dataset saved to {output_file}")
        print(f"Processed {len(dataset)} entries, saved {len(cleaned_data)} valid entries")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def load_dataset(file_path: str) -> List[Dict]:
    """
    Load dataset from a JSONL file.

    Args:
        file_path (str): Path to the input JSONL file

    Returns:
        List[Dict]: List of JSON objects from the file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def process_entries(dataset: List[Dict]) -> List[Dict]:
    """
    Process and validate each entry in the dataset.

    Args:
        dataset (List[Dict]): List of data entries to process

    Returns:
        List[Dict]: List of validated and cleaned entries
    """
    cleaned_data = []
    
    for entry in dataset:
        # Verify required fields exist
        if not all(key in entry for key in ('InputText', 'SentimentLabel')):
            print(f"Warning: Missing required fields in entry: {entry}")
            continue

        # Validate SentimentLabel is a string
        if not isinstance(entry['SentimentLabel'], str):
            print(f"Warning: Invalid SentimentLabel type in entry: {entry}")
            continue

        # Validate InputText is not empty
        if not entry['InputText'].strip():
            print(f"Warning: Empty InputText in entry: {entry}")
            continue

        cleaned_data.append(entry)
    
    return cleaned_data

def save_dataset(data: List[Dict], file_path: str) -> None:
    """
    Save the cleaned dataset to a JSONL file.

    Args:
        data (List[Dict]): Cleaned data to save
        file_path (str): Path where the data will be saved
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    # Example usage
    input_file = 'csvtojsondata.jsonl'
    output_file = 'data.jsonl'
    clean_dataset(input_file, output_file)