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

def compare_parquet_files(symbol, base_dir='./data'):
    """
    Compare the Parquet files for the given symbol from Alpaca and yfinance data sources.
    """
    data_sources = ['alpaca', 'yfinance']
    data_frames = {}

    for data_source in data_sources:
        data_dir = Path(base_dir) / data_source
        file_path = data_dir / f"{symbol}.parquet"

        if not file_path.exists():
            logging.warning(f"No Parquet file found for symbol {symbol} in {data_source}")
            continue

        logging.info(f"\nReading Parquet file for symbol {symbol} from {data_source}: {file_path}")
        try:
            df = pd.read_parquet(file_path)
            data_frames[data_source] = df

            # Display basic information about the DataFrame
            logging.debug(f"\nData from {data_source} for {symbol}:")
            logging.debug(f"Columns: {df.columns.tolist()}")
            logging.debug(f"Index names: {df.index.names}")
            logging.debug(f"Index type: {type(df.index)}")
            logging.debug(f"Index dtype(s): {df.index.dtypes if hasattr(df.index, 'dtypes') else df.index.dtype}")
            logging.debug(f"Data types:\n{df.dtypes}")
            logging.debug(f"Head of DataFrame:\n{df.head()}")
            logging.debug(f"Tail of DataFrame:\n{df.tail()}")
        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")

    if len(data_frames) != 2:
        logging.warning(f"Could not load data for symbol {symbol} from both data sources.")
        return

    # Compare columns
    columns_match = data_frames['alpaca'].columns.equals(data_frames['yfinance'].columns)
    logging.info(f"\nColumns match: {columns_match}")
    if not columns_match:
        logging.info(f"Alpaca columns: {data_frames['alpaca'].columns.tolist()}")
        logging.info(f"yfinance columns: {data_frames['yfinance'].columns.tolist()}")

    # Compare index names
    index_names_match = data_frames['alpaca'].index.names == data_frames['yfinance'].index.names
    logging.info(f"Index names match: {index_names_match}")
    if not index_names_match:
        logging.info(f"Alpaca index names: {data_frames['alpaca'].index.names}")
        logging.info(f"yfinance index names: {data_frames['yfinance'].index.names}")

    # Compare index types
    index_types_match = type(data_frames['alpaca'].index) == type(data_frames['yfinance'].index)
    logging.info(f"Index types match: {index_types_match}")
    if not index_types_match:
        logging.info(f"Alpaca index type: {type(data_frames['alpaca'].index)}")
        logging.info(f"yfinance index type: {type(data_frames['yfinance'].index)}")

        # Get index dtypes
    alpaca_index = data_frames['alpaca'].index
    yfinance_index = data_frames['yfinance'].index

    if isinstance(alpaca_index, pd.MultiIndex):
        alpaca_index_dtypes = alpaca_index.dtypes
    else:
        alpaca_index_dtypes = pd.Series([alpaca_index.dtype], index=[alpaca_index.name])

    if isinstance(yfinance_index, pd.MultiIndex):
        yfinance_index_dtypes = yfinance_index.dtypes
    else:
        yfinance_index_dtypes = pd.Series([yfinance_index.dtype], index=[yfinance_index.name])

    # Compare index dtypes
    index_dtypes_match = alpaca_index_dtypes.equals(yfinance_index_dtypes)

    logging.info(f"Index dtypes match: {index_dtypes_match}")
    if not index_dtypes_match:
        logging.info(f"Alpaca index dtypes:\n{alpaca_index_dtypes}")
        logging.info(f"yfinance index dtypes:\n{yfinance_index_dtypes}")

    # Compare data types
    dtypes_match = data_frames['alpaca'].dtypes.equals(data_frames['yfinance'].dtypes)
    logging.info(f"Data types match: {dtypes_match}")
    if not dtypes_match:
        logging.info(f"Alpaca data types:\n{data_frames['alpaca'].dtypes}")
        logging.info(f"yfinance data types:\n{data_frames['yfinance'].dtypes}")

    # Compare sample data (optional)
    # You can implement additional comparisons such as checking for matching dates, values, etc.

def main(base_dir='./data', symbol=None):
    """
    Main function to execute the testing script.
    """
    if symbol:
        compare_parquet_files(symbol, base_dir=base_dir)
    else:
        logging.error("Please provide a symbol using the --symbol argument.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Parquet files for a symbol between Alpaca and yfinance.")
    parser.add_argument('--base-dir', default='./data', help="Base directory for data.")
    parser.add_argument('--symbol', required=True, help="Symbol to compare.")
    args = parser.parse_args()

    # Execute the main function
    main(args.base_dir, args.symbol)
