import pandas as pd
import numpy as np

def load_core_asset_returns(stock_returns_path, bond_returns_path, name_to_ticker_map):
    """
    Loads stock and bond returns, merges them, and applies ticker mapping.
    """
    stock_returns = pd.read_csv(stock_returns_path, index_col=0, parse_dates=True)
    bond_returns = pd.read_csv(bond_returns_path, index_col=0, parse_dates=True)

    # Notebook cleaning
    if "b1ret" in bond_returns.columns:
        bond_returns = bond_returns.drop(columns=["b1ret"])
    if "b7ret" in bond_returns.columns:
        bond_returns = bond_returns.drop(columns=["b7ret"])

    bond_returns.columns.name = None
    bond_returns.index.name = None
    bond_returns.index = pd.to_datetime(bond_returns.index)
    bond_returns = bond_returns.apply(pd.to_numeric)

    stock_returns_c = stock_returns.copy()
    bond_returns_c = bond_returns.copy()

    stock_returns_c = stock_returns_c.reset_index().rename(columns={'index': 'Date', 'Date': 'Date'})
    bond_returns_c = bond_returns_c.reset_index().rename(columns={'index': 'Date', 'Date': 'Date'})

    # Ensure 'Date' is datetime
    stock_returns_c['Date'] = pd.to_datetime(stock_returns_c['Date'])
    bond_returns_c['Date'] = pd.to_datetime(bond_returns_c['Date'])

    returns_df = pd.merge(stock_returns_c, bond_returns_c, on='Date', how='inner')
    returns_df.set_index('Date', inplace=True)
    returns_df.index.name = None

    returns_df.rename(columns=name_to_ticker_map, inplace=True)
    return returns_df

def load_benchmark_returns(benchmark_path, returns_index):
    """
    Loads benchmark returns and aligns them to the asset returns index.
    """
    sp_benchmark = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
    sp_benchmark = sp_benchmark.loc[returns_index]
    return sp_benchmark["sprtrn"] # 'sprtrn' is benchmark col

def load_risk_free_rates(rf_path, desired_start_date, desired_end_date, returns_index):
    """
    Loads, processes, and aligns risk-free rate data.
    """
    rf = pd.read_csv(rf_path)
    rf.columns = ['Date', 'Yield']
    rf['Date'] = pd.to_datetime(rf['Date'])
    rf.set_index('Date', inplace=True)
    rf = rf.dropna()

    rf_monthly = rf.resample('M').last()
    rf_monthly_shifted = rf_monthly.shift(1)
    # Ann. yield to monthly return
    # Original: continuous compounding
    # Simpler: (yield/100)/12
    rf_monthly_shifted['rf_return'] = (rf_monthly_shifted['Yield'] / 100) / 12


    rf_final = rf_monthly_shifted.loc[desired_start_date:desired_end_date]
    rf_final_aligned = rf_final.reindex(returns_index, method='ffill')
    return rf_final_aligned['rf_return']

def calculate_excess_returns(asset_returns_df, risk_free_series):
    """
    Computes excess returns for each asset.
    """
    excess_returns_df = asset_returns_df.sub(risk_free_series, axis=0)
    return excess_returns_df

def load_historical_asset_returns_for_prior(historical_returns_path, target_columns, start_row=24):
    """
    Loads historical asset returns (e.g., pre-2008) for Bayesian priors.
    Assumes target_columns are the columns to keep from the loaded CSV,
    matching the main period's asset names.
    """
    historical_df = pd.read_csv(historical_returns_path, index_col=0, parse_dates=True)
    if start_row is not None:
        historical_df = historical_df.iloc[start_row:].copy()

    # Keep common cols (hist & target)
    columns_to_keep = [col for col in target_columns if col in historical_df.columns]
    historical_df = historical_df[columns_to_keep]
    return historical_df

def load_historical_risk_free_for_prior(rf_path, desired_start_date, desired_end_date, historical_returns_index):
    """
    Loads and processes historical risk-free rates for the pre-sample period.
    """
    # Similar to load_risk_free_rates
    rf_hist = pd.read_csv(rf_path)
    rf_hist.columns = ['Date', 'Yield']
    rf_hist['Date'] = pd.to_datetime(rf_hist['Date'])
    rf_hist.set_index('Date', inplace=True)
    rf_hist = rf_hist.dropna()

    rf_hist_monthly = rf_hist.resample('M').last()
    rf_hist_monthly_shifted = rf_hist_monthly.shift(1)
    rf_hist_monthly_shifted['rf_return'] = (rf_hist_monthly_shifted['Yield'] / 100) / 12


    rf_hist_final = rf_hist_monthly_shifted.loc[desired_start_date:desired_end_date]
    rf_hist_final_aligned = rf_hist_final.reindex(historical_returns_index, method='ffill')
    return rf_hist_final_aligned['rf_return']

def load_historical_benchmark_for_prior(benchmark_pre_path, desired_start_date, desired_end_date):
    """
    Loads and filters historical S&P 500 returns for the pre-sample period.
    """
    sp500_pre = pd.read_csv(benchmark_pre_path, index_col=0, parse_dates=True)
    sp500_pre_filtered = sp500_pre.loc[desired_start_date:desired_end_date]
    sp500_pre_filtered = sp500_pre_filtered.rename(columns={'sprtrn': 'sp500_return'}) # Assume 'sprtrn' col
    return sp500_pre_filtered['sp500_return']
