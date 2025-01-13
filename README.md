# Asset Prices to Parquet - Price Data Saver

## Overview
The **Asset Prices to Parquet** toolkit is a comprehensive solution for handling **historical** and **live** financial price data.

Robust backtesting is key to any quantitative strategy's success. Reliable historical and live data are essential for evaluating and refining your approach. Continuously querying APIs during development and testing can lead to slowdowns and increased costs. This program addresses these pain points by downloading symbol data from **Alpaca** and **Yahoo Finance** and storing it locally in the efficient **Parquet** format. By using local data files, you can test your strategies at scale without worrying about API limits, latency, or connectivity issues.

- **Data Sources**: Alpaca API and Yahoo Finance (YFinance).
- **Data Format**: Prices are stored in the efficient Parquet format.
- **Functionality**:
  - Initial bulk downloads for historical data.
  - Real-time live price updates for active and tradable assets.

This repository is ideal for **financial analysts**, **quantitative researchers**, and **algorithmic traders** seeking a robust price data management solution.

---

## Features

### 1. Historical Data Management (`download_historical_price.py`)
- **Sources**: Retrieves data from **Alpaca** and **Yahoo Finance**.
- **Storage**: Saves adjusted close prices in `.parquet` format for easy and efficient use.
- **Invalid Symbol Handling**: Automatically identifies and skips invalid or delisted symbols.
- **Automatic Updates**: Supports scheduled updates for the entire dataset or specific symbols.

### 2. Live Data Management (`update_live_price.py`)
- **Live Price Updates**: Retrieves the latest trade price from **Alpaca**.
- **Batch Processing**: Efficiently handles large datasets using batch API requests.
- **Dynamic Invalid Symbols Management**: Detects and skips invalid symbols automatically during polling.

---

## Project Structure

```
.
├── data/
│   ├── alpaca/          # Historical and live price data from Alpaca
│   ├── yfinance/        # Historical price data from Yahoo Finance
│   ├── logs/            # Logs for all operations
│   ├── yfinance_invalid_symbols.txt
│   └── alpaca_invalid_symbols.txt
├── compare_parquet.py   # Compares Alpaca and YFinance Parquet files for a symbol.
├── parquet_details.py   # Displays schema, head, and tail of Parquet files.
└── download_historical_price.py  # Handles historical data downloads and updates.
└── update_live_price.py  # Polls live prices and updates Parquet files.
```

---

## Installation

### Prerequisites

1. **Python Version**: Python 3.8 or later.

2. **Dependencies**: Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies:
   - `alpaca-trade-api`
   - `yfinance`
   - `pandas`
   - `requests`
   - `python-dotenv`

3. **Environment Variables**:
   Create a `.env` file with your Alpaca API credentials:

   ```
   ALPACA_API_KEY=<your_api_key>
   ALPACA_API_SECRET=<your_api_secret>
   ```

---

## Usage

### 1. Download Historical Data

- **Initial Download**: Download data for all symbols:
  ```bash
  python download_historical_price.py --initial-download
  ```
- **Update Specific Symbol**:
  ```bash
  python download_historical_price.py --symbol AAPL
  ```

### 2. Update Live Prices

- Start live price updates:
  ```bash
  python update_live_price.py
  ```

---

## Key Functions

### Comparison and Validation

- Compare data files:
  ```bash
  python compare_parquet.py --symbol AAPL
  ```
- Display head, tail, and schema:
  ```bash
  python parquet_details.py --symbol AAPL
  ```

---

## Logging and Error Handling

### Logs

All activities are logged in the `data/logs/` directory:

- Successful operations
- Skipped invalid symbols
- Error reports

### Invalid Symbols

Invalid symbols are saved in separate files (`yfinance_invalid_symbols.txt` and `alpaca_invalid_symbols.txt`).

---

## Contribution

Contributions are welcome! Submit a pull request or create an issue to suggest improvements or report bugs.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **Alpaca API** for trading and historical data.
- **Yahoo Finance** for its accessible financial data service.

---

## GitHub Description

**Comprehensive toolkit for saving historical and live financial price data from Alpaca and Yahoo Finance into Parquet files.**

---

Copy and paste this format directly into your GitHub `README.md` for a ready-to-use project description.

