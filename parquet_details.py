import os
import sys
import pandas as pd
from pathlib import Path
import argparse
import logging

# Configure logging for the script
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_latest_parquet_files(data_dir, num_files=5):
    """
    Retrieve the latest modified Parquet files from the specified directory.
    """
    logging.debug(f"Retrieving the latest {num_files} Parquet files from {data_dir}")
    parquet_files = list(Path(data_dir).glob("*.parquet"))
    # Sort files by modified time in descending order
    sorted_files = sorted(parquet_files, key=lambda f: f.stat().st_mtime, reverse=True)
    return sorted_files[:num_files]

def show_head_tail_and_schema(file_paths, num_rows=5):
    """
    Display the head, tail, and schema of each specified Parquet file.
    """
    for file_path in file_paths:
        logging.info(f"Displaying head, tail, and schema of {file_path}")
        try:
            df = pd.read_parquet(file_path)
            print(f"\nHead of {file_path}:\n", df.head(num_rows))
            print(f"\nTail of {file_path}:\n", df.tail(num_rows))
            print(f"\nSchema of {file_path}:\n", df.dtypes)
            print(f"\nColumn names in {file_path}:\n", df.columns.tolist())
        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")

def main(base_dir='./data', num_files=5, num_rows=5, symbol=None):
    """
    Main function to execute the testing script.
    """
    data_sources = ['alpaca', 'yfinance']
    for data_source in data_sources:
        data_dir = Path(base_dir) / data_source
        if not data_dir.exists() or not data_dir.is_dir():
            logging.error(f"The specified data directory {data_dir} does not exist.")
            continue

        logging.info(f"\nProcessing data source: {data_source}")

        if symbol:
            # Look for Parquet file of the specified symbol
            file_path = data_dir / f"{symbol}.parquet"
            if file_path.exists():
                logging.info(f"Testing Parquet file for symbol {symbol}: {file_path}")
                show_head_tail_and_schema([file_path], num_rows=num_rows)
            else:
                logging.warning(f"No Parquet file found for symbol {symbol} in {data_source}")
        else:
            # Retrieve the latest Parquet files
            latest_files = get_latest_parquet_files(data_dir, num_files=num_files)
            # Display the head, tail, and schema of each file
            show_head_tail_and_schema(latest_files, num_rows=num_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the latest Parquet files in alpaca and yfinance.")
    parser.add_argument('--base-dir', default='./data', help="Base directory for data.")
    parser.add_argument('--num-files', type=int, default=5, help="Number of latest files to show.")
    parser.add_argument('--num-rows', type=int, default=5,
                        help="Number of rows to display from each file's head and tail.")
    parser.add_argument('--symbol', help="Specific symbol to test. If provided, only this symbol will be tested.")
    args = parser.parse_args()

    # Execute the main function
    main(args.base_dir, args.num_files, args.num_rows, args.symbol)
