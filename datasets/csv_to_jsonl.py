import csv
import json
from typing import List, Dict, Optional

def convert_csv_to_jsonl(csv_file: str, jsonl_file: str, text_column: int = 1, label_column: int = 2) -> Optional[int]:
    """
    Convert a CSV file to JSONL format for sentiment analysis.
    
    Args:
        csv_file (str): Path to the input CSV file
        jsonl_file (str): Path where the JSONL output will be saved
        text_column (int): Index of the column containing the text data (default: 1)
        label_column (int): Index of the column containing the sentiment labels (default: 2)
    
    Returns:
        Optional[int]: Number of records processed successfully, or None if an error occurred
    
    The function expects:
    - CSV file with a header row
    - Text column containing the input text for sentiment analysis
    - Label column containing numeric sentiment values (0 or 1)
    """
    processed_count = 0

    try:
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            # Validate CSV file structure
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)  # Read header row
            
            if len(header) <= max(text_column, label_column):
                raise ValueError(f"CSV file doesn't have enough columns. Expected at least {max(text_column, label_column) + 1} columns")

            processed_count = process_csv_rows(csv_reader, jsonl_file, text_column, label_column)
            print(f"Successfully converted {processed_count} records to JSONL format")
            return processed_count

    except FileNotFoundError:
        print(f"Error: Input file '{csv_file}' not found")
    except csv.Error as e:
        print(f"Error reading CSV file: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    
    return None

def process_csv_rows(csv_reader: csv.reader, jsonl_file: str, text_column: int, label_column: int) -> int:
    """
    Process CSV rows and write them to JSONL format.
    
    Args:
        csv_reader: CSV reader object
        jsonl_file (str): Output JSONL file path
        text_column (int): Index of the text column
        label_column (int): Index of the label column
    
    Returns:
        int: Number of records processed successfully
    """
    processed_count = 0
    errors = []

    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for row_num, row in enumerate(csv_reader, start=2):  # Start from 2 to account for header
            try:
                # Validate row structure
                if len(row) <= max(text_column, label_column):
                    errors.append(f"Row {row_num}: Invalid number of columns")
                    continue

                # Validate and clean text
                text = row[text_column].strip()
                if not text:
                    errors.append(f"Row {row_num}: Empty text field")
                    continue

                # Validate and convert sentiment label
                label = validate_sentiment_label(row[label_column], row_num)
                if label is None:
                    continue

                # Create and write JSON record
                json_record = {
                    "InputText": text,
                    "SentimentLabel": label
                }
                jsonlfile.write(json.dumps(json_record, ensure_ascii=False) + '\n')
                processed_count += 1

            except Exception as e:
                errors.append(f"Row {row_num}: Error processing row: {str(e)}")

    # Report errors if any occurred
    if errors:
        print("\nWarnings during conversion:")
        for error in errors[:10]:  # Show first 10 errors
            print(error)
        if len(errors) > 10:
            print(f"...and {len(errors) - 10} more warnings")

    return processed_count

def validate_sentiment_label(label_str: str, row_num: int) -> Optional[str]:
    """
    Validate and convert sentiment label to required format.
    
    Args:
        label_str (str): Raw label value from CSV
        row_num (int): Current row number for error reporting
    
    Returns:
        Optional[str]: Validated label as string ('0' or '1') or None if invalid
    """
    try:
        label_int = int(label_str)
        if label_int not in (0, 1):
            print(f"Row {row_num}: Invalid sentiment label value: {label_str} (must be 0 or 1)")
            return None
        return str(label_int)
    except ValueError:
        print(f"Row {row_num}: Invalid sentiment label format: {label_str}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = 'TuniziDataset.csv'
    output_file = 'csvtojsondata.jsonl'
    
    print(f"Converting {input_file} to JSONL format...")
    records_processed = convert_csv_to_jsonl(input_file, output_file)
    
    if records_processed:
        print(f"Conversion completed: {records_processed} records written to {output_file}")