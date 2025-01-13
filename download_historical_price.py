#!/usr/bin/env python3
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
from pathlib import Path
import logging
import asyncio
import argparse
import time
import sys

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import yfinance as yf
from dotenv import load_dotenv

# -----------------------------------------------------------------------
# Section: Market cap integration from backtest_pairs_v2.py
# -----------------------------------------------------------------------
from importlib.util import spec_from_file_location, module_from_spec

def load_market_cap_module():
    """
    Dynamically loads the market cap module (get_market_cap.py)
    from a known file path. Adjust as needed for your environment.
    """
    eodhd_path = "/home/shared/algos/eodhd/tools/get_market_cap.py"
    spec = spec_from_file_location("get_market_cap", eodhd_path)
    market_cap_module = module_from_spec(spec)
    spec.loader.exec_module(market_cap_module)
    return market_cap_module

# -----------------------------------------------------------------------

load_dotenv()

# Configure logging at the start of the script
logging.basicConfig(
    level=logging.WARNING,  # Set to DEBUG level if needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout
    ]
)


class HistoricalDataManager:
    def __init__(self, data_dir='/home/shared/algos/asset_prices/data', initial_download=False):
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
        self.yfinance_data_dir = self.data_dir / 'yfinance'
        self.logs_dir = self.data_dir / 'logs'

        # Create directories if they don't exist
        self.alpaca_data_dir.mkdir(parents=True, exist_ok=True)
        self.yfinance_data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize invalid_symbols as empty sets
        self.invalid_yfinance_symbols = set()
        self.invalid_alpaca_symbols = set()

        # Load invalid symbols from separate files
        self.load_invalid_symbols('yfinance_invalid_symbols.txt', 'yfinance')
        self.load_invalid_symbols('alpaca_invalid_symbols.txt', 'alpaca')

        # Initialize Alpaca's Trading Client to fetch assets
        logging.debug("Initializing Alpaca Trading Client")
        self.trading_client = TradingClient(
            api_key=self.alpaca_api_key,
            secret_key=self.alpaca_api_secret,
            paper=False
        )

        # Initialize Alpaca's Historical Data Client
        logging.debug("Initializing Alpaca Historical Data Client")
        self.historical_client = StockHistoricalDataClient(
            api_key=self.alpaca_api_key,
            secret_key=self.alpaca_api_secret
        )

        # We will NOT automatically set self.symbols = get_active_symbols() here
        # We'll select them later, based on user arguments.
        self.symbols = []

        # Handle initial historical data download
        self.initial_download = initial_download
        self.next_download_time = None

    def load_invalid_symbols(self, invalid_symbols_file, source):
        """Load invalid symbols from a file for a specific source."""
        file_path = self.data_dir / invalid_symbols_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                symbols = set(f.read().splitlines())
            if source == 'yfinance':
                self.invalid_yfinance_symbols = symbols
            elif source == 'alpaca':
                self.invalid_alpaca_symbols = symbols
            logging.info(f"Loaded {len(symbols)} invalid symbols for {source} from {invalid_symbols_file}.")
        else:
            logging.info(f"No invalid symbols file found at {file_path} for {source}.")

    def add_invalid_yfinance_symbol(self, symbol):
        """Add a symbol to the list of invalid YFinance symbols and save it."""
        if symbol not in self.invalid_yfinance_symbols:
            self.invalid_yfinance_symbols.add(symbol)
            file_path = self.data_dir / 'yfinance_invalid_symbols.txt'
            with open(file_path, 'a') as f:
                f.write(f"{symbol}\n")
            logging.info(f"Added {symbol} to the invalid YFinance symbols list.")

    def add_invalid_alpaca_symbol(self, symbol):
        """Add a symbol to the list of invalid Alpaca symbols and save it."""
        if symbol not in self.invalid_alpaca_symbols:
            self.invalid_alpaca_symbols.add(symbol)
            file_path = self.data_dir / 'alpaca_invalid_symbols.txt'
            with open(file_path, 'a') as f:
                f.write(f"{symbol}\n")
            logging.info(f"Added {symbol} to the invalid Alpaca symbols list.")

    def get_active_symbols(self):
        """
        Fetch all 'active' symbols from Alpaca, regardless of tradeability.
        """
        logging.debug("Entering get_active_symbols method")
        try:
            search_params = GetAssetsRequest(status='active')
            assets = self.trading_client.get_all_assets(search_params)
            # Remove tradable filter: fetch all active symbols
            symbols = [asset.symbol for asset in assets]
            logging.info(f"Fetched {len(symbols)} active symbols from Alpaca (including non-tradable).")
            return symbols
        except Exception as e:
            logging.error(f"Error fetching active symbols from Alpaca: {e}")
            return []

    def fetch_and_combine_symbols(self, args):
        """
        Gathers symbols from:
          - market cap categories (if specified, including "all" for all categories)
          - active/tradable from Alpaca (if requested)
          - single symbol (if specified via --symbol)
        Then sets self.symbols to the union of all.
        """
        final_symbols = set()

        # 1) Handle market cap categories
        if args.categories:
            try:
                market_cap_module = load_market_cap_module()
                market_caps_list = market_cap_module.get_market_caps(
                    data_dir="/home/shared/algos/eodhd/data/fundamental_data/",
                    output_csv="/home/shared/algos/eodhd/data/market_caps.csv"
                )
                categories_dict = market_cap_module.categorize_market_caps(dict(market_caps_list))

                if "all" in args.categories:
                    # Include all categories if "all" is specified
                    for category_symbols in categories_dict.values():
                        final_symbols.update(category_symbols)
                else:
                    # Include only the specified categories
                    for cat in args.categories:
                        cat_lower = cat.lower()
                        if cat_lower in categories_dict:
                            final_symbols.update(categories_dict[cat_lower])
                        else:
                            logging.warning(f"Category '{cat}' not recognized by get_market_cap.")
            except Exception as e:
                logging.error(f"Failed to fetch market cap categories: {e}")

        # 2) If user included --include-active, get all active/tradable from Alpaca
        if args.include_active:
            final_symbols.update(self.get_active_symbols())

        # 3) If user specified a single symbol with --symbol, just add it
        if args.symbol:
            final_symbols.add(args.symbol.upper())  # uppercase to be consistent

        # If the user didn't specify anything, default to all active & tradable
        if not final_symbols:
            logging.info("No symbols specified; defaulting to all active & tradable from Alpaca.")
            final_symbols = set(self.get_active_symbols())

        self.symbols = sorted(final_symbols)
        logging.info(f"Total final symbol list: {len(self.symbols)}")

    def run(self):
        """
        Main method to run the historical data manager.
        - If initial_download is set, download full historical data right away.
        - Then schedule daily downloads.
        """
        logging.debug("Running main HistoricalDataManager method")
        if self.initial_download:
            logging.info("Performing initial historical data download")
            asyncio.run(self.download_all_historical_data())
            self.initial_download = False

        self.schedule_daily_download()
        asyncio.run(self.main())

    def schedule_daily_download(self):
        """Set up the daily download schedule for one hour after market close."""
        now = datetime.now()
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        self.next_download_time = market_close + timedelta(hours=1)

        if self.next_download_time < now:
            self.next_download_time += timedelta(days=1)

        logging.info(f"Next scheduled download time: {self.next_download_time}")

    async def main(self):
        """Asynchronous main loop for scheduling data downloads."""
        logging.debug("Starting main async loop for historical data")
        while True:
            now = datetime.now()
            if self.next_download_time and now >= self.next_download_time:
                logging.info("Scheduled time reached, downloading historical data")
                await self.download_all_historical_data()
                self.schedule_daily_download()

            logging.debug("Historical data main loop sleeping for 1 hour")
            await asyncio.sleep(3600)  # Sleep for an hour, repeat

    async def download_all_historical_data(self):
        """
        Download full historical data for all symbols from both Alpaca
        and yfinance. If the user did `--symbol X`, we only update that one.
        """
        # If we *only* want to update a single symbol, check if self.symbols has length 1, etc.
        if len(self.symbols) == 1:
            symbol = self.symbols[0]
            logging.info(f"Updating single symbol: {symbol}")
            await self.update_single_symbol_historical_data(symbol)
        else:
            logging.info("Starting download of Alpaca historical data.")
            await asyncio.to_thread(self.download_full_historical_data_alpaca)
            logging.info("Completed download of Alpaca historical data.")

            logging.info("Starting download of YFinance historical data sequentially.")
            await asyncio.to_thread(self.download_full_historical_data_yfinance_sequential)
            logging.info("Completed download of YFinance historical data.")

    def download_full_historical_data_alpaca(self, batch_size=10, delay_between_batches=2):
        """Download full historical adjusted data for all symbols from Alpaca."""
        logging.info("Starting download of full historical data from Alpaca...")

        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]
            for symbol in batch:
                self.fetch_and_save_historical_data_alpaca(symbol)
            logging.debug(f"Completed batch {i // batch_size + 1} of Alpaca historical data download.")
            time.sleep(delay_between_batches)

        logging.info("Completed download of historical data from Alpaca.")

    def download_full_historical_data_yfinance_sequential(self):
        """
        Download full historical adjusted data for all symbols from Yahoo Finance,
        one at a time.
        """
        logging.info("Starting sequential download of historical data from Yahoo Finance...")

        for symbol in self.symbols:
            if symbol in self.invalid_yfinance_symbols:
                logging.info(f"Skipping {symbol} (marked invalid for YFinance).")
                continue
            self.update_symbol_data_yfinance_sequential(symbol)
            time.sleep(1)  # Add delay if desired to respect API rate limits

        logging.info("Completed sequential download of historical data from Yahoo Finance.")

    def update_symbol(self, symbol):
        """Update historical data for a specific symbol from both Alpaca and YFinance."""
        asyncio.run(self.update_single_symbol_historical_data(symbol))

    async def update_single_symbol_historical_data(self, symbol):
        """Asynchronously update data for a single symbol from both sources."""
        logging.info(f"Starting data update for symbol: {symbol}")
        await asyncio.gather(
            asyncio.to_thread(self.fetch_and_save_historical_data_alpaca, symbol),
            asyncio.to_thread(self.update_symbol_data_yfinance_sequential, symbol)
        )
        logging.info(f"Completed data update for symbol: {symbol}")

    def fetch_and_save_historical_data_alpaca(self, symbol,
                                              symbols_without_data_file='alpaca_symbols_without_data.txt'):
        """
        Fetch and save historical data from Alpaca with adjusted close prices.
        """
        if symbol in self.invalid_alpaca_symbols:
            logging.info(f"Skipping {symbol} (marked invalid for Alpaca).")
            return

        # Convert symbols with '-' to '.' for compatibility with Alpaca
        alpaca_symbol = symbol.replace('-', '.')
        logging.debug(f"Fetching historical data for Alpaca symbol: {alpaca_symbol}")

        try:
            symbols_without_data_path = self.data_dir / symbols_without_data_file
            if symbols_without_data_path.exists():
                with open(symbols_without_data_path, 'r') as f:
                    symbols_without_data = set(f.read().splitlines())
            else:
                symbols_without_data = set()

            if alpaca_symbol in symbols_without_data:
                logging.info(f"Skipping {alpaca_symbol} (found in alpaca_symbols_without_data.txt).")
                return

            request_params = StockBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                adjustment="all",
                timeframe=TimeFrame.Day,
                start=datetime(1970, 1, 1, tzinfo=timezone.utc),
                end=datetime.now(timezone.utc),
                feed="sip",
            )
            try:
                bars = self.historical_client.get_stock_bars(request_params)
                df = bars.df.copy()

                if df.empty:
                    logging.warning(f"No Alpaca historical data found for {alpaca_symbol}.")
                    symbols_without_data.add(alpaca_symbol)
                    with open(symbols_without_data_path, 'w') as f:
                        f.write('\n'.join(sorted(symbols_without_data)))
                    return

                df.rename(columns={'close': 'adj_close'}, inplace=True)
                df.reset_index(inplace=True)
                df.set_index('timestamp', inplace=True)
                df.drop(columns=['symbol'], inplace=True)

                # Save to Parquet
                file_path = self.alpaca_data_dir / f"{alpaca_symbol}.parquet"
                temp_file_path = file_path.with_suffix('.parquet.tmp')
                df.to_parquet(temp_file_path)
                os.replace(temp_file_path, file_path)

                logging.info(f"Saved historical Alpaca data for {alpaca_symbol} to {file_path} "
                             f"with adjusted 'adj_close' column.")

            except Exception as e:
                logging.warning(f"Failed to fetch data for {alpaca_symbol}. Error: {e}")
                symbols_without_data.add(alpaca_symbol)
                with open(symbols_without_data_path, 'w') as f:
                    f.write('\n'.join(sorted(symbols_without_data)))

        except Exception as e:
            logging.error(f"Error fetching/saving Alpaca data for {alpaca_symbol}: {e}")

    def update_symbol_data_yfinance_sequential(self, symbol):
        """
        Fetch and save historical data from YFinance sequentially, ensuring no mix-ups.
        """
        if symbol in self.invalid_yfinance_symbols:
            logging.info(f"Skipping {symbol} (marked invalid for YFinance).")
            return

        logging.debug(f"Fetching historical data for {symbol} from Yahoo Finance")
        file_path = self.yfinance_data_dir / f"{symbol}.parquet"
        temp_file_path = file_path.with_suffix('.parquet.tmp')

        # Replace dots with hyphens for compatibility with YFinance
        yf_symbol = symbol.replace('.', '-')
        logging.info(f"Using YFinance symbol: {yf_symbol}")

        try:
            bars = yf.download(
                yf_symbol,
                interval='1d',
                progress=False,
                threads=False  # Disable threading in yfinance
            )

            # If no data from the yf_symbol, try the original symbol
            if bars.empty:
                logging.warning(f"No YFinance data for {yf_symbol}. Trying original symbol.")
                bars = yf.download(
                    symbol,
                    interval='1d',
                    progress=False,
                    threads=False
                )

            if bars.empty:
                logging.warning(f"No YFinance data for {symbol}. Marking as possibly delisted.")
                self.add_invalid_yfinance_symbol(symbol)
                return

            logging.info(f"Successfully fetched data for {symbol} from Yahoo Finance")

        except Exception as e:
            logging.error(f"Unexpected error fetching YFinance data for {symbol}: {e}")
            return

        # Flatten MultiIndex columns if present and remove the symbol name
        if isinstance(bars.columns, pd.MultiIndex):
            logging.debug(f"Data for {symbol} has MultiIndex columns: {bars.columns}")
            if 'Ticker' in bars.columns.names:
                bars.columns = bars.columns.droplevel('Ticker')
            else:
                bars.columns = bars.columns.droplevel(level=1)
        else:
            logging.debug(f"Data for {symbol} has single-level columns: {bars.columns}")

        # Convert all column names to lowercase
        bars.columns = [col.lower() for col in bars.columns]
        # Rename 'adj close' to 'adj_close'
        if 'adj close' in bars.columns:
            bars.rename(columns={'adj close': 'adj_close'}, inplace=True)

        # Convert index to datetime and set timezone to UTC
        bars.index = pd.to_datetime(bars.index)
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize('UTC')
        else:
            bars.index = bars.index.tz_convert('UTC')
        bars.index.rename('timestamp', inplace=True)

        # Save to Parquet
        try:
            bars.to_parquet(temp_file_path)
            os.replace(temp_file_path, file_path)
            logging.info(f"Saved YFinance data for {symbol} to {file_path} in an Alpaca-compatible format.")
        except Exception as e:
            logging.error(f"Error writing YFinance data for {symbol}: {e}")

# -----------------------------------------------------------------------
# Argument parsing & main entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HistoricalDataManager for asset prices.")
    parser.add_argument('--initial-download', action='store_true',
                        help="Download all historical data upon startup.")

    # Single symbol override (optional)
    parser.add_argument('--symbol', type=str, help="Specify a single symbol to update its historical data only.")

    # Market-cap category integration:
    # You can specify multiple categories at once (e.g. --categories mega large mid)
    parser.add_argument('--categories', nargs='*',
                        help="List of market cap categories to include (e.g. mega, large, mid, small, micro, nano, unknown, all)")

    # Include Alpaca's active & tradable symbols
    parser.add_argument('--include-active', action='store_true',
                        help="Include all Alpaca symbols that are 'active' & 'tradable' in the download set.")

    args = parser.parse_args()

    logging.debug("Starting HistoricalDataManager with given arguments")

    data_manager = HistoricalDataManager(initial_download=args.initial_download)
    # Gather final symbol list from user arguments
    data_manager.fetch_and_combine_symbols(args)

    # If user specifically wants to update only one symbol, do that; else run normally
    if args.symbol and len(data_manager.symbols) == 1:
        logging.info(f"Updating data for specific symbol: {args.symbol}")
        data_manager.update_symbol(args.symbol)
    else:
        # Run the manager to update data for all gathered symbols
        data_manager.run()
