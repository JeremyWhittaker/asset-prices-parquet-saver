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

import requests
import json

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import yfinance as yf
from tqdm import tqdm
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

        self.eodhd_api_key = os.getenv('EODHD_API_KEY')  # needed for EODHD

        if not self.alpaca_api_key or not self.alpaca_api_secret:
            raise ValueError("Please set ALPACA_API_KEY and ALPACA_API_SECRET as environment variables.")

        if not self.eodhd_api_key:
            logging.warning("EODHD_API_KEY not set. EODHD symbol fetching will fail if used.")

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

        # We'll set self.symbols later, based on user arguments
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

    def get_active_symbols_alpaca(self):
        """
        Fetch all 'active' symbols from Alpaca (regardless of tradeability).
        """
        logging.debug("Entering get_active_symbols_alpaca method")
        try:
            search_params = GetAssetsRequest(status='active')
            assets = self.trading_client.get_all_assets(search_params)
            symbols = [asset.symbol for asset in assets]
            logging.info(f"Fetched {len(symbols)} 'active' symbols from Alpaca.")
            return symbols
        except Exception as e:
            logging.error(f"Error fetching active symbols from Alpaca: {e}")
            return []

    # ---------------------------------------------------------------------
    # EODHD approach: This is a simplified version of #get_symbols_from_exchange.py
    # Instead of saving to JSON, we just return all symbols from relevant exchanges.
    # If you want advanced filtering or to handle certain exchanges only, adjust logic here.
    # ---------------------------------------------------------------------
    def get_symbols_from_eodhd(self, exchange_codes=None):
        """
        Fetch symbols from EODHD for given exchange_codes. If none provided,
        fetch from a default set or from all possible exchanges.
        """
        if not self.eodhd_api_key:
            logging.error("EODHD_API_KEY not available; cannot fetch from EODHD.")
            return []

        if exchange_codes is None:
            # Provide a default list if you only want certain exchanges,
            # or fetch the entire list from EODHD (might be huge).
            exchange_codes = ["NYSE", "NASDAQ", "BATS", "ARCA"]  # Example subset

        all_symbols = set()
        for exch in exchange_codes:
            try:
                symbols_data = self._fetch_symbols_for_exchange(exch)
                # Each entry typically has { "Code": "AAPL", "Name": ..., "Exchange": "NASDAQ", ... }
                # Just gather the "Code" as the symbol
                for item in symbols_data:
                    symbol = item["Code"].upper()
                    all_symbols.add(symbol)
            except Exception as e:
                logging.warning(f"Failed to fetch symbols for exchange {exch}: {e}")

        logging.info(f"Total EODHD symbols gathered: {len(all_symbols)} across {exchange_codes}")
        return sorted(all_symbols)

    def _fetch_symbols_for_exchange(self, exchange_code):
        """
        Fetch the list of symbols for a given exchange via EODHD.
        """
        base_url = "https://eodhd.com/api"
        url = f"{base_url}/exchange-symbol-list/{exchange_code}?api_token={self.eodhd_api_key}&fmt=json"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


    def fetch_and_combine_symbols(self, args):
        """
        Gathers symbols from the specified source (EODHD or Alpaca),
        market cap categories (if specified), and single symbol (if specified).
        Then sets self.symbols to the combined set.
        """
        final_symbols = set()

        # 1) Choose symbol source based on argument
        if args.symbols_source == 'alpaca':
            logging.info("Using Alpaca to fetch active symbols based on --symbols-source alpaca.")
            final_symbols.update(self.get_active_symbols_alpaca())
        else:
            logging.info("Using EODHD to fetch symbols based on --symbols-source eodhd.")
            final_symbols.update(self.get_symbols_from_eodhd())

        # 2) Handle market cap categories, if specified
        if args.categories:
            try:
                market_cap_module = load_market_cap_module()
                market_caps_list = market_cap_module.get_market_caps(
                    data_dir="/home/shared/algos/eodhd/data/fundamental_data/",
                    output_csv="/home/shared/algos/eodhd/data/market_caps.csv"
                )
                categories_dict = market_cap_module.categorize_market_caps(dict(market_caps_list))

                if "all" in args.categories:
                    for category_symbols in categories_dict.values():
                        final_symbols.update(category_symbols)
                else:
                    for cat in args.categories:
                        cat_lower = cat.lower()
                        if cat_lower in categories_dict:
                            final_symbols.update(categories_dict[cat_lower])
                        else:
                            logging.warning(f"Category '{cat}' not recognized by get_market_cap.")
            except Exception as e:
                logging.error(f"Failed to fetch market cap categories: {e}")

        # 3) If user specified a single symbol with --symbol, add it
        if args.symbol:
            final_symbols.add(args.symbol.upper())

        # Warn if no symbols found
        if not final_symbols:
            logging.warning("No symbols found from the chosen sources. The final symbol set is empty.")

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
        Download historical data for all symbols from the selected data source(s).
        If data_source == 'all', it downloads from both Alpaca & YFinance in parallel.
        """
        if len(self.symbols) == 1:
            # If there's exactly one symbol, do the single-symbol update method
            symbol = self.symbols[0]
            logging.info(f"Updating single symbol: {symbol}")
            await self.update_single_symbol_historical_data(symbol)
            return

        if self.data_source == "all":
            logging.info("Downloading data from BOTH Alpaca & YFinance (--data-source=all).")
            # Run both in parallel
            await asyncio.gather(
                asyncio.to_thread(self.download_full_historical_data_alpaca),
                asyncio.to_thread(self.download_full_historical_data_yfinance_sequential)
            )
        elif self.data_source == "alpaca":
            logging.info("Downloading data ONLY from Alpaca (--data-source=alpaca).")
            await asyncio.to_thread(self.download_full_historical_data_alpaca)
        elif self.data_source == "yfinance":
            logging.info("Downloading data ONLY from YFinance (--data-source=yfinance).")
            await asyncio.to_thread(self.download_full_historical_data_yfinance_sequential)

        logging.info("Completed download_all_historical_data()")

    def download_full_historical_data_alpaca(self, batch_size=10, delay_between_batches=2):
        """
        Download full historical data from Alpaca in batches,
        now with a tqdm progress bar.
        """
        logging.info("Starting Alpaca download.")
        total_syms = len(self.symbols)
        with tqdm(total=total_syms, desc="Alpaca Download") as pbar:
            for i in range(0, total_syms, batch_size):
                batch = self.symbols[i : i + batch_size]
                for symbol in batch:
                    self.fetch_and_save_historical_data_alpaca(symbol)
                    pbar.update(1)  # increment the bar by 1
                time.sleep(delay_between_batches)
        logging.info("Completed Alpaca download.")

    def download_full_historical_data_yfinance_sequential(self):
        """
        Download full historical data from YFinance, one symbol at a time,
        with a tqdm progress bar.
        """
        logging.info("Starting YFinance download.")
        total_syms = len(self.symbols)
        with tqdm(total=total_syms, desc="YFinance Download") as pbar:
            for symbol in self.symbols:
                if symbol in self.invalid_yfinance_symbols:
                    logging.info(f"Skipping {symbol} (invalid for YFinance).")
                else:
                    self.update_symbol_data_yfinance_sequential(symbol)
                pbar.update(1)
                time.sleep(0.5)  # optional short delay to avoid hitting rate limits
        logging.info("Completed YFinance download.")

    def update_symbol(self, symbol):
        """Update historical data for a specific symbol from both Alpaca and YFinance."""
        asyncio.run(self.update_single_symbol_historical_data(symbol))

    async def update_single_symbol_historical_data(self, symbol):
        """
        Update data for a single symbol from the selected data source(s).
        """
        logging.info(f"Starting data update for symbol: {symbol}")

        if self.data_source == "all":
            # Update from both sources in parallel
            await asyncio.gather(
                asyncio.to_thread(self.fetch_and_save_historical_data_alpaca, symbol),
                asyncio.to_thread(self.update_symbol_data_yfinance_sequential, symbol)
            )
        elif self.data_source == "alpaca":
            # Update ONLY from Alpaca
            await asyncio.to_thread(self.fetch_and_save_historical_data_alpaca, symbol)
        elif self.data_source == "yfinance":
            # Update ONLY from YFinance
            await asyncio.to_thread(self.update_symbol_data_yfinance_sequential, symbol)

        logging.info(f"Completed data update for symbol: {symbol}")

    def fetch_and_save_historical_data_alpaca(self, symbol,
                                              symbols_without_data_file='alpaca_symbols_without_data.txt'):
        """
        Fetch and save historical data from Alpaca with adjusted close prices.
        """
        if symbol in self.invalid_alpaca_symbols:
            logging.info(f"Skipping {symbol} (marked invalid for Alpaca).")
            return

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

        # Convert column names to lowercase
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HistoricalDataManager for asset prices.")
    parser.add_argument('--initial-download', action='store_true',
                        help="Download all historical data upon startup.")
    parser.add_argument('--symbol', type=str,
                        help="Specify a single symbol to update its historical data only.")
    parser.add_argument('--categories', nargs='*',
                        help="List of market cap categories (e.g. mega, large, mid, small, etc.)")

    # NEW ARGUMENT: --data-source
    parser.add_argument('--data-source', choices=['all', 'alpaca', 'yfinance'], default='all',
                        help="Specify which data source(s) to download. Options: all, alpaca, yfinance. Default is 'all'.")

    # Keep the existing --symbols-source or anything else you have:
    parser.add_argument('--symbols-source', choices=['alpaca', 'eodhd'], default='eodhd',
                        help="Specify the source for fetching symbols. Default is eodhd.")

    args = parser.parse_args()

    data_manager = HistoricalDataManager(initial_download=args.initial_download)
    # Collect symbols first (this is unchanged)
    data_manager.fetch_and_combine_symbols(args)

    # Store the chosen data source in the data_manager so we can check it later
    data_manager.data_source = args.data_source

    if args.symbol and len(data_manager.symbols) == 1:
        data_manager.update_symbol(args.symbol)
    else:
        data_manager.run()
