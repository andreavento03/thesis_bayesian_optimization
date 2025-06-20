import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration (will be defined in the main block or a config file later) ---
# Example: RAW_COMPANIES_2008_PATH = '/Users/andreavento/Documents/Tesi/Dataset/03_01_2008_companies.csv'

# --- Function Definitions ---

def process_top_companies_2008(input_csv_path, output_plot_path=None):
    """
    Loads company data, calculates market cap, identifies top 20, and optionally plots.
    """
    df = pd.read_csv(input_csv_path)

    df['PRC'] = pd.to_numeric(df['PRC'], errors='coerce')
    df['SHROUT'] = pd.to_numeric(df['SHROUT'], errors='coerce')
    df['MarketCap'] = df['PRC'] * df['SHROUT']

    top_20_market_cap = df.sort_values(by='MarketCap', ascending=False).head(20)
    top_20_companies_df = top_20_market_cap[['COMNAM', 'MarketCap']].copy() # Use .copy() to avoid SettingWithCopyWarning

    if output_plot_path: # In a script, directly showing plot might not be ideal unless specified
        plt.figure(figsize=(12, 6))
        plt.bar(top_20_companies_df['COMNAM'], top_20_companies_df['MarketCap'])
        plt.xticks(rotation=90)
        plt.xlabel('Company')
        plt.ylabel('Market Capitalization (USD)')
        plt.title('Top 20 Companies by Market Cap (as of file date)')
        plt.tight_layout()
        plt.savefig(output_plot_path) # Save the plot
        plt.close() # Close the plot to free memory
        print(f"Saved top companies plot to {output_plot_path}")

    return top_20_companies_df

def process_main_stock_returns(input_csv_path, output_csv_path, company_name_map):
    """
    Loads, cleans, and pivots main stock return data.
    """
    returns_df = pd.read_csv(input_csv_path)

    returns_df['COMNAM'] = returns_df['COMNAM'].replace(company_name_map)
    returns_clean = returns_df.groupby(['date', 'COMNAM'])['RET'].mean().reset_index()
    returns_wide = returns_clean.pivot(index='date', columns='COMNAM', values='RET').sort_index()

    returns_wide.columns.name = None
    returns_wide.index.name = None
    returns_wide.index = pd.to_datetime(returns_wide.index)
    returns_wide = returns_wide.apply(pd.to_numeric)

    returns_wide.to_csv(output_csv_path, index=True)
    return returns_wide

def process_bitcoin_returns(input_csv_path):
    """
    Loads and processes Bitcoin data to get monthly returns.
    """
    btc_daily_df = pd.read_csv(input_csv_path)
    btc_daily_df["Date"] = pd.to_datetime(btc_daily_df["Date"], format="%m/%d/%Y")
    # Assuming 'Price' column might have commas for thousands
    if btc_daily_df["Price"].dtype == 'object':
        btc_daily_df["Price"] = btc_daily_df["Price"].str.replace(",", "", regex=False).astype(float)
    else:
        btc_daily_df["Price"] = btc_daily_df["Price"].astype(float)


    btc_daily_df = btc_daily_df.sort_values("Date")
    btc_daily_df.set_index("Date", inplace=True)

    btc_monthly_price = btc_daily_df["Price"].resample("M").last()
    btc_monthly_return = btc_monthly_price.pct_change()
    btc_returns_df = btc_monthly_return.to_frame(name="btc_ret") # Ensure correct naming

    btc_returns_df.columns.name = None
    btc_returns_df.index.name = None
    return btc_returns_df

def merge_stock_and_bitcoin_returns(stocks_df, bitcoin_df, output_csv_path):
    """
    Merges stock returns with Bitcoin returns.
    """
    # Ensure indices are compatible (e.g., both are Timestamps at month end)
    stocks_df_copy = stocks_df.copy()
    bitcoin_df_copy = bitcoin_df.copy()

    stocks_df_copy.index = pd.to_datetime(stocks_df_copy.index).to_period("M").to_timestamp("M")
    bitcoin_df_copy.index = pd.to_datetime(bitcoin_df_copy.index).to_period("M").to_timestamp("M")

    merged_df = stocks_df_copy.join(bitcoin_df_copy, how="left")
    merged_df.to_csv(output_csv_path, index=True)
    return merged_df

def process_historical_stock_returns(input_csv_path, ticker_map, num_rows=None):
    """
    Loads, cleans, and pivots historical stock return data.
    """
    old_returns_df = pd.read_csv(input_csv_path)
    old_returns_df['Standardized Ticker'] = old_returns_df['COMNAM'].map(ticker_map)
    old_returns_df.dropna(subset=['Standardized Ticker'], inplace=True)
    old_returns_df['RET'] = pd.to_numeric(old_returns_df['RET'], errors='coerce')
    old_returns_df = old_returns_df.groupby(['date', 'Standardized Ticker'])['RET'].mean().reset_index()
    old_returns_df_pivot = old_returns_df.pivot(index='date', columns='Standardized Ticker', values='RET').sort_index()

    if num_rows is not None:
        old_returns_df_pivot = old_returns_df_pivot.iloc[:num_rows].copy() # Use .iloc for positional slicing

    old_returns_df_pivot.columns.name = None
    old_returns_df_pivot.index.name = None
    old_returns_df_pivot.index = pd.to_datetime(old_returns_df_pivot.index) # Ensure datetime index
    return old_returns_df_pivot

def process_historical_bond_returns(input_csv_path, num_rows=None):
    """
    Loads and cleans historical bond return data.
    """
    old_returns_bond_df = pd.read_csv(input_csv_path)
    # Drop specific columns if they exist
    cols_to_drop = ["b1ret", "b7ret"]
    old_returns_bond_df.drop(columns=[col for col in cols_to_drop if col in old_returns_bond_df.columns], inplace=True)

    if num_rows is not None:
        old_returns_bond_df = old_returns_bond_df.iloc[:num_rows].copy()

    old_returns_bond_df.rename(columns={'caldt': 'Date', 'b20ret': 'BOND20'}, inplace=True)
    old_returns_bond_df.set_index('Date', inplace=True)
    old_returns_bond_df.index = pd.to_datetime(old_returns_bond_df.index) # Ensure datetime index
    return old_returns_bond_df

def merge_historical_returns(hist_stocks_df, hist_bonds_df, output_csv_path):
    """
    Merges historical stock and bond returns.
    """
    # Ensure indices are compatible before merging
    stock_returns_reset = hist_stocks_df.reset_index().rename(columns={'index': 'Date', 'date': 'Date'})
    bond_returns_reset = hist_bonds_df.reset_index().rename(columns={'index': 'Date', 'Date': 'Date'})

    # Ensure 'Date' column is datetime
    stock_returns_reset['Date'] = pd.to_datetime(stock_returns_reset['Date'])
    bond_returns_reset['Date'] = pd.to_datetime(bond_returns_reset['Date'])

    merged_df = pd.merge(stock_returns_reset, bond_returns_reset, on='Date', how='inner')
    merged_df.set_index('Date', inplace=True)
    merged_df.index.name = None

    merged_df.to_csv(output_csv_path, index=True)
    return merged_df

# Main execution block will be added in the next step.
# print("Refactored dataset_generation.py with function definitions.")


# --- Configuration ---
# Note: User will need to update these paths if they differ.
BASE_DATA_PATH = '/Users/andreavento/Documents/Tesi/Dataset/' # Base path for convenience

RAW_COMPANIES_2008_PATH = BASE_DATA_PATH + '03_01_2008_companies.csv'
PLOT_TOP_COMPANIES_PATH = BASE_DATA_PATH + 'top_20_companies_market_cap_2008.png' # Example output path for plot

RAW_STOCKS_UNCLEANED_PATH = BASE_DATA_PATH + 'Final Dataset/final_dataset_stocks_ucleaned.csv'
FINAL_STOCKS_PATH = BASE_DATA_PATH + 'Final Dataset/final_dataset_stocks.csv'

BITCOIN_DATA_PATH = BASE_DATA_PATH + 'Bitcoin Historical Data-2.csv'
FINAL_STOCKS_WITH_BITCOIN_PATH = BASE_DATA_PATH + 'Final Dataset/final_dataset_stocks_with_bitcoin.csv'

OLD_STOCK_PATH = BASE_DATA_PATH + 'pre-estimation-capm/old_stock.csv'
OLD_BOND_PATH = BASE_DATA_PATH + 'pre-estimation-capm/old_bond.csv'
HISTORICAL_COMBINED_PATH = BASE_DATA_PATH + 'pre-estimation-capm/historical_before2008_returns.csv'

COMPANY_NAME_NORMALIZATION_MAP = {
    'GOOGLE INC': 'ALPHABET INC',
    'WAL MART STORES INC': 'WALMART INC'
}

HISTORICAL_TICKER_MAP = {
    'A T & T INC': 'T', 'S B C COMMUNICATIONS INC': 'T', 'ALPHABET INC': 'GOOGL',
    'ALTRIA GROUP INC': 'MO', 'PHILIP MORRIS COS INC': 'MO', 'AMERICAN INTERNATIONAL GROUP INC': 'AIG',
    'APPLE INC': 'AAPL', 'BANK OF AMERICA CORP': 'BAC', 'BERKSHIRE HATHAWAY INC DEL': 'BRK.B',
    'CHEVRON CORP NEW': 'CVX', 'CHEVRON CORP': 'CVX', 'CHEVRONTEXACO CORP': 'CVX',
    'CISCO SYSTEMS INC': 'CSCO', 'CITIGROUP INC': 'C', 'COCA COLA CO': 'KO',
    'EXXON MOBIL CORP': 'XOM', 'GENERAL ELECTRIC CO': 'GE', 'INTEL CORP': 'INTC',
    'INTERNATIONAL BUSINESS MACHS COR': 'IBM', 'JOHNSON & JOHNSON': 'JNJ',
    'MICROSOFT CORP': 'MSFT', 'PFIZER INC': 'PFE', 'PROCTER & GAMBLE CO': 'PG',
    'WALMART INC': 'WMT', 'APPLE COMPUTER INC': 'AAPL', 'AT&T INC': 'T',
    'ALPHABET INC CL C': 'GOOGL', 'ALPHABET INC CL A': 'GOOGL', 'GOOGLE INC': 'GOOGL',
    'BERKSHIRE HATHAWAY INC. DEL': 'BRK.B', 'EXXON MOBIL CORP.': 'XOM',
    'GENERAL ELECTRIC CO.': 'GE', 'INTEL CORP.': 'INTC',
    'INTERNATIONAL BUSINESS MACHS CORP': 'IBM', 'JOHNSON & JOHNSON.': 'JNJ',
    'MICROSOFT CORP.': 'MSFT', 'PFIZER INC.': 'PFE', 'PROCTER & GAMBLE CO.': 'PG',
    'WAL-MART STORES INC': 'WMT', 'WAL MART STORES INC': 'WMT'
}
NUM_HISTORICAL_ROWS = 96


# --- Main Execution Workflow ---
if __name__ == "__main__":
    print("Starting dataset generation...")

    # Process top 2008 companies (generates a plot)
    print("\nProcessing top 20 companies from 2008...")
    _ = process_top_companies_2008(RAW_COMPANIES_2008_PATH, PLOT_TOP_COMPANIES_PATH)
    print(f"Processed top 20 companies and saved plot to {PLOT_TOP_COMPANIES_PATH}")

    # Process main stock returns
    print("\nProcessing main stock returns...")
    main_stocks_df = process_main_stock_returns(
        RAW_STOCKS_UNCLEANED_PATH,
        FINAL_STOCKS_PATH,
        COMPANY_NAME_NORMALIZATION_MAP
    )
    print(f"Saved processed main stock returns to {FINAL_STOCKS_PATH}")

    # Process Bitcoin returns
    print("\nProcessing Bitcoin returns...")
    btc_df = process_bitcoin_returns(BITCOIN_DATA_PATH)
    print("Processed Bitcoin monthly returns.")

    # Merge stocks and Bitcoin
    print("\nMerging main stock returns with Bitcoin returns...")
    _ = merge_stock_and_bitcoin_returns(
        main_stocks_df,
        btc_df,
        FINAL_STOCKS_WITH_BITCOIN_PATH
    )
    print(f"Saved merged stock and Bitcoin returns to {FINAL_STOCKS_WITH_BITCOIN_PATH}")

    # Process historical stock returns
    print("\nProcessing historical stock returns (pre-2008)...")
    hist_stocks_df = process_historical_stock_returns(
        OLD_STOCK_PATH,
        HISTORICAL_TICKER_MAP,
        NUM_HISTORICAL_ROWS
    )
    print("Processed historical stock returns.")

    # Process historical bond returns
    print("\nProcessing historical bond returns (pre-2008)...")
    hist_bonds_df = process_historical_bond_returns(OLD_BOND_PATH, NUM_HISTORICAL_ROWS)
    print("Processed historical bond returns.")

    # Merge historical returns
    print("\nMerging historical stock and bond returns...")
    _ = merge_historical_returns(
        hist_stocks_df,
        hist_bonds_df,
        HISTORICAL_COMBINED_PATH
    )
    print(f"Saved combined historical returns to {HISTORICAL_COMBINED_PATH}")

    print("\nDataset generation complete.")
