import os
from datetime import datetime, timezone, timedelta
import pandas as pd
from pathlib import Path
import logging
import threading
import time
import sys
import requests
import argparse
import numpy as np

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest

from dotenv import load_dotenv
load_dotenv()

# Configure logging at the start of the script
logging.basicConfig(
    level=logging.WARNING,  # Set to DEBUG level if needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout
    ]
)

class LiveDataManager:
    def __init__(self, data_dir='/home/shared/algos/asset_prices/data'):
        # Retrieve API credentials from environment variables
        logging.debug("Retrieving API credentials")
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_api_secret = os.getenv('ALPACA_API_SECRET')

        if not self.alpaca_api_key or not self.alpaca_api_secret:
            raise ValueError("Please set ALPACA_API_KEY and ALPACA_API_SECRET as environment variables.")

        # Initialize data directories
        logging.debug("Initializing data directories")
        self.data_dir = Path(data_dir)
        self.alpaca_data_dir = self.data_dir / 'alpaca'
        self.logs_dir = self.data_dir / 'logs'

        # Create directories if they don't exist
        self.alpaca_data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize invalid_symbols as an empty set
        self.invalid_alpaca_symbols = set()

        # Load invalid symbols from file
        self.load_invalid_symbols('alpaca_invalid_symbols.txt', 'alpaca')

        # Initialize Alpaca's Trading Client to fetch assets
        logging.debug("Initializing Alpaca Trading Client")
        self.trading_client = TradingClient(
            api_key=self.alpaca_api_key,
            secret_key=self.alpaca_api_secret,
            paper=False
        )

        # Fetch all active and tradable symbols from Alpaca
        logging.debug("Fetching all active and tradable symbols from Alpaca")
        self.symbols = self.get_active_symbols()
        logging.info(f"Total active and tradable symbols: {len(self.symbols)}")

        # Initialize an empty DataFrame to hold current prices
        logging.debug("Initializing empty DataFrame for current prices")
        self.current_prices = pd.DataFrame(index=self.symbols, columns=['price'], dtype=float)

    def load_invalid_symbols(self, invalid_symbols_file, source):
        """Load invalid symbols from a file for a specific source."""
        file_path = self.data_dir / invalid_symbols_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                symbols = set(f.read().splitlines())
            if source == 'alpaca':
                self.invalid_alpaca_symbols = symbols
            logging.info(f"Loaded {len(symbols)} invalid symbols for {source} from {invalid_symbols_file}.")
        else:
            logging.info(f"No invalid symbols file found at {file_path} for {source}.")

    def add_invalid_alpaca_symbol(self, symbol):
        """Add a symbol to the list of invalid Alpaca symbols and save it."""
        if symbol not in self.invalid_alpaca_symbols:
            self.invalid_alpaca_symbols.add(symbol)
            file_path = self.data_dir / 'alpaca_invalid_symbols.txt'
            with open(file_path, 'a') as f:
                f.write(f"{symbol}\n")
            logging.info(f"Added {symbol} to the invalid Alpaca symbols list.")

    def get_active_symbols(self):
        """Fetch all active and tradable symbols from Alpaca."""
        logging.debug("Entering get_active_symbols method")
        try:
            search_params = GetAssetsRequest(status='active')
            assets = self.trading_client.get_all_assets(search_params)
            symbols = [asset.symbol for asset in assets if asset.tradable]
            logging.info(f"Fetched {len(symbols)} active and tradable symbols.")
            return symbols
        except Exception as e:
            logging.error(f"Error fetching active symbols from Alpaca: {e}")
            return []

    def run(self):
        """Main method to run the live data manager."""
        logging.debug("Running main LiveDataManager method")
        self.start_latest_trades_polling()

        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Exiting...")
            sys.exit(0)

    def start_latest_trades_polling(self, batch_size=100, polling_interval=60):
        """
        Start polling the latest trades data for Alpaca symbols in batches.
        """
        logging.debug("Starting latest trades polling")

        def poll_latest_trades():
            try:
                url = "https://data.alpaca.markets/v2/stocks/trades/latest"
                headers = {
                    "accept": "application/json",
                    "APCA-API-KEY-ID": self.alpaca_api_key,
                    "APCA-API-SECRET-KEY": self.alpaca_api_secret
                }

                # Poll the latest trades for all valid symbols in batches
                while True:
                    logging.info("Polling latest trades for all valid Alpaca symbols in batches...")
                    valid_symbols = [symbol for symbol in self.symbols if symbol not in self.invalid_alpaca_symbols]

                    # Log the number of valid symbols being processed
                    logging.debug(f"Number of valid symbols: {len(valid_symbols)}")

                    # Process symbols in batches
                    for i in range(0, len(valid_symbols), batch_size):
                        batch = valid_symbols[i:i + batch_size]
                        symbols_str = ",".join(batch)  # Join the batch of symbols into a single string

                        logging.debug(f"Processing batch {i // batch_size + 1}: {batch}")

                        try:
                            params = {"symbols": symbols_str, "feed": "sip"}
                            response = requests.get(url, headers=headers, params=params)

                            if response.status_code == 200:
                                trades_data = response.json().get("trades", {})
                                # Process the trades data for each symbol
                                for symbol, trade in trades_data.items():
                                    price = trade.get("p")
                                    timestamp = pd.Timestamp(trade.get("t"))

                                    # Check if price and timestamp are valid
                                    if price is not None and not pd.isna(
                                            price) and timestamp is not None and not pd.isna(timestamp):
                                        logging.debug(
                                            f"Date/time: {timestamp}: Received latest trade for {symbol} with price {price}")
                                        self.current_prices.at[symbol, 'price'] = price
                                        self.update_parquet_with_live_price(symbol, price, timestamp)
                                    else:
                                        logging.warning(
                                            f"No valid trade data for {symbol}. Price: {price}, Timestamp: {timestamp}")

                            elif response.status_code == 400:
                                # Handle invalid symbols
                                message = response.json().get("message", "")
                                invalid_symbols = []
                                if "the following symbols were invalid" in message:
                                    # Extract invalid symbols from the message
                                    invalid_symbols = message.split(":")[1].strip().split(", ")
                                    for invalid_symbol in invalid_symbols:
                                        logging.error(f"Invalid symbol detected: {invalid_symbol}")
                                        self.add_invalid_alpaca_symbol(invalid_symbol)
                                else:
                                    logging.error(f"Error response: {message}")

                            else:
                                logging.error(
                                    f"Failed to fetch latest trades. Status code: {response.status_code}, Response: {response.text}")

                        except Exception as e:
                            logging.error(f"Error fetching latest trades: {e}")

                    # Wait for the polling interval before repeating
                    logging.debug(f"Sleeping for {polling_interval} seconds before next polling cycle")
                    time.sleep(polling_interval)
            except Exception as e:
                logging.error(f"Exception in poll_latest_trades: {e}", exc_info=True)

        # Run polling in a separate thread
        polling_thread = threading.Thread(target=poll_latest_trades, daemon=True)
        polling_thread.start()

    def update_parquet_with_live_price(self, symbol, price, timestamp):
        """Update the Parquet file with the latest price, using the index as the timestamp."""
        logging.debug(f"Starting update for {symbol}: new price {price} at {timestamp}")
        file_path = self.alpaca_data_dir / f"{symbol}.parquet"

        # Format the timestamp to match the existing rows and ensure timezone is UTC
        formatted_timestamp = pd.Timestamp(timestamp).floor('s').tz_convert('UTC')

        if not file_path.exists():
            logging.warning(f"Parquet file for {symbol} does not exist. Creating a new file.")
            # Create a new DataFrame with index as `timestamp` and consistent columns
            df = pd.DataFrame(
                {
                    'open': [np.nan],
                    'high': [np.nan],
                    'low': [np.nan],
                    'adj_close': [float(price)],
                    'volume': [np.nan],
                    'trade_count': [np.nan],
                    'vwap': [np.nan]
                },
                index=[formatted_timestamp]
            )
            df.index.name = 'timestamp'  # Set the index name to `timestamp`
        else:
            try:
                # Read the existing Parquet file
                df = pd.read_parquet(file_path)
                logging.debug(f"Successfully read Parquet file for {symbol}. Number of rows: {len(df)}")
                logging.debug(f"DataFrame index name: {df.index.name}")
                logging.debug(f"DataFrame columns: {df.columns}")

                # Ensure the index is named 'timestamp' and is timezone-aware in UTC
                if df.index.name != 'timestamp':
                    if 'timestamp' in df.columns:
                        df.reset_index(inplace=True)
                        df.set_index('timestamp', inplace=True)
                    else:
                        df.index.name = 'timestamp'
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')

                # Ensure all required columns are present
                required_columns = ['open', 'high', 'low', 'adj_close', 'volume', 'trade_count', 'vwap']
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = np.nan

                # Normalize timestamps to dates for comparison
                existing_dates = df.index.normalize()
                current_date = formatted_timestamp.normalize()

                if current_date in existing_dates:
                    # Update the 'adj_close' price for the existing date
                    date_mask = existing_dates == current_date
                    df.loc[date_mask, 'adj_close'] = float(price)
                    logging.debug(f"Updated existing row for {symbol} on date {current_date.date()}")
                else:
                    # Append a new row with the current price
                    new_row = pd.DataFrame(
                        {
                            'open': [np.nan],
                            'high': [np.nan],
                            'low': [np.nan],
                            'adj_close': [float(price)],
                            'volume': [np.nan],
                            'trade_count': [np.nan],
                            'vwap': [np.nan]
                        },
                        index=[formatted_timestamp]
                    )
                    df = pd.concat([df, new_row])
                    logging.debug(f"Appended new row for {symbol} on date {current_date.date()}")

            except Exception as e:
                logging.error(
                    f"Error reading Parquet file for {symbol}. Price: {price}, Timestamp: {formatted_timestamp}. Error: {e}"
                )
                return

        # Sort the DataFrame by index to maintain chronological order
        df.sort_index(inplace=True)

        # Ensure index name is 'timestamp'
        df.index.name = 'timestamp'

        # Save back to the Parquet file using a temporary file for atomic write
        temp_file_path = file_path.with_suffix('.parquet.tmp')
        try:
            df.to_parquet(temp_file_path)
            os.replace(temp_file_path, file_path)
            logging.info(f"Updated Parquet file for {symbol} with latest price at {formatted_timestamp}.")
        except Exception as e:
            logging.error(
                f"Error writing Parquet file for {symbol}. Price: {price}, Timestamp: {formatted_timestamp}. Error: {e}"
            )


if __name__ == "__main__":
    logging.debug("Starting LiveDataManager")
    data_manager = LiveDataManager()
    data_manager.run()
