import pandas as pd
import numpy as np
# Assumes portfolio_optimizers and bayesian_estimators are accessible.
# Relative imports for main script context.

# Placeholder: solve_portfolio (if not directly imported)
DEFAULT_SOLVER_FALLBACK_WEIGHTS = None

def solve_portfolio_placeholder(mu, Sigma, risk_limit_sq, constraints_fn, ridge=False, lambda_ridge=0.1):
    print(f"Placeholder: solve_portfolio called for {constraints_fn.__name__}")
    n = len(mu)
    if DEFAULT_SOLVER_FALLBACK_WEIGHTS == 'equal':
        return np.full(n, 1/n)
    return np.random.rand(n) # Random weights fallback


def run_rolling_strategy(
    excess_returns_df,
    estimation_window,
    holding_period,
    risk_limit_target_vol, # Target vol (e.g., 0.03 monthly)
    constraint_fn,
    portfolio_solver_fn,
    ridge=False,
    lambda_ridge=0.1
):
    """
    Runs a rolling window backtest for classical MVO strategies.
    risk_limit_target_vol: Target monthly volatility (e.g., 0.03).
                           This will be squared to get risk_limit_sq for the solver.
    portfolio_solver_fn: The function to call for solving portfolio weights, e.g., solve_portfolio.
    """
    weights_over_time = []
    portfolio_returns_list = []
    portfolio_return_dates = []
    rebalance_dates = []
    T = len(excess_returns_df)
    asset_names_master_list = excess_returns_df.columns.tolist() # Use original assets for consistent df structure
    risk_limit_sq_val = risk_limit_target_vol**2

    for start_idx in range(0, T - estimation_window - holding_period + 1, holding_period):
        est_data = excess_returns_df.iloc[start_idx : start_idx + estimation_window].dropna(axis=1, how='all')

        if est_data.empty or est_data.shape[1] == 0:
            print(f"Skipping period starting at {excess_returns_df.index[start_idx]} due to no valid assets after dropna in estimation window.")
            continue

        current_asset_names = est_data.columns.tolist()

        hold_data = excess_returns_df.iloc[start_idx + estimation_window : start_idx + estimation_window + holding_period]
        hold_data_aligned = hold_data[current_asset_names] # Align cols to available assets
        current_hold_dates = hold_data_aligned.index

        mu_hat = est_data.mean().values
        Sigma_hat = est_data.cov().values

        if Sigma_hat.shape[0] == 0: # Skip if no assets after dropna
            print(f"Skipping period due to empty Sigma_hat at {excess_returns_df.index[start_idx + estimation_window]}")
            continue

        weights = portfolio_solver_fn(mu_hat, Sigma_hat, risk_limit_sq_val, constraint_fn, ridge=ridge, lambda_ridge=lambda_ridge)

        if weights is not None:
            port_rets = hold_data_aligned.dot(weights)
            portfolio_returns_list.extend(port_rets)
            portfolio_return_dates.extend(current_hold_dates)
            # Store weights Series, align later
            weights_series = pd.Series(weights, index=current_asset_names)
            weights_over_time.append(weights_series)
            rebalance_dates.append(current_hold_dates[0] if not current_hold_dates.empty else pd.NaT)
        else:
            print(f"Warning: Weights could not be computed for period starting {excess_returns_df.index[start_idx + estimation_window]}")

    if not rebalance_dates or pd.isna(rebalance_dates).all():
        print("Warning: No valid rebalance dates found. Returning empty results.")
        return pd.DataFrame(columns=asset_names_master_list), pd.Series(dtype=float)

    weights_df = pd.DataFrame(weights_over_time, index=rebalance_dates)
    weights_df = weights_df.reindex(columns=asset_names_master_list).fillna(0) # Align cols, fill NaNs

    if not portfolio_returns_list:
        return weights_df, pd.Series(dtype=float, name="Rolling Portfolio Excess Returns")

    returns_series = pd.Series(portfolio_returns_list, name="Rolling Portfolio Excess Returns", index=portfolio_return_dates)
    return weights_df, returns_series

def run_bayesian_strategy(
    excess_returns_df,
    estimation_window,
    holding_period,
    risk_limit_target_vol,
    constraint_fn,
    posterior_fn, # posterior_fn: returns (mu, Sigma)
    portfolio_solver_fn,
    ridge=False,
    lambda_ridge=0.1,
    **posterior_kwargs
):
    """
    Runs a rolling window backtest for Bayesian strategies that do NOT require benchmark data for posterior.
    posterior_fn: e.g., bayesian_diffuse_prior_moments, informative_bayes_predictive_niw (with priors fixed or from config)
    """
    weights_over_time = []
    portfolio_returns_list = []
    portfolio_return_dates = []
    rebalance_dates = []
    T = len(excess_returns_df)
    asset_names_master_list = excess_returns_df.columns.tolist()
    risk_limit_sq_val = risk_limit_target_vol**2

    for start_idx in range(0, T - estimation_window - holding_period + 1, holding_period):
        est_data = excess_returns_df.iloc[start_idx : start_idx + estimation_window].dropna(axis=1, how='all')

        if est_data.empty or est_data.shape[1] == 0:
            print(f"Skipping period at {excess_returns_df.index[start_idx]} due to no valid assets in estimation data.")
            continue

        current_asset_names = est_data.columns.tolist()

        hold_data = excess_returns_df.iloc[start_idx + estimation_window : start_idx + estimation_window + holding_period]
        hold_data_aligned = hold_data[current_asset_names]
        current_hold_dates = hold_data_aligned.index

        # Calc posterior moments
        mu_posterior, Sigma_posterior = posterior_fn(est_data, **posterior_kwargs)

        if Sigma_posterior.shape[0] == 0:
            print(f"Skipping period due to empty Sigma_posterior at {excess_returns_df.index[start_idx + estimation_window]}")
            continue

        weights = portfolio_solver_fn(mu_posterior, Sigma_posterior, risk_limit_sq_val, constraint_fn, ridge=ridge, lambda_ridge=lambda_ridge)

        if weights is not None:
            port_rets = hold_data_aligned.dot(weights)
            portfolio_returns_list.extend(port_rets)
            portfolio_return_dates.extend(current_hold_dates)
            weights_series = pd.Series(weights, index=current_asset_names)
            weights_over_time.append(weights_series)
            rebalance_dates.append(current_hold_dates[0] if not current_hold_dates.empty else pd.NaT)
        else:
            print(f"Warning: Weights could not be computed for Bayesian strategy for period starting {excess_returns_df.index[start_idx + estimation_window]}")

    if not rebalance_dates or pd.isna(rebalance_dates).all():
        print("Warning: No valid rebalance dates for Bayesian strategy. Returning empty results.")
        return pd.DataFrame(columns=asset_names_master_list), pd.Series(dtype=float)

    weights_df = pd.DataFrame(weights_over_time, index=rebalance_dates)
    weights_df = weights_df.reindex(columns=asset_names_master_list).fillna(0)

    if not portfolio_returns_list:
        return weights_df, pd.Series(dtype=float, name="Bayesian Portfolio Excess Returns")

    returns_series = pd.Series(portfolio_returns_list, name="Bayesian Portfolio Excess Returns", index=portfolio_return_dates)
    return weights_df, returns_series

def run_custom_bayesian_strategy_with_benchmark(
    excess_returns_df,
    benchmark_returns_series, # Benchmark returns for posterior (e.g., B-CAPM)
    estimation_window,
    holding_period,
    risk_limit_target_vol,
    constraint_fn,
    posterior_fn,
    portfolio_solver_fn,
    gmv_vol_calculator_fn=None, # Optional: GMV func for adaptive risk
    ridge=False,
    lambda_ridge=0.1,
    adaptive_risk_target_vol=None, # Adaptive risk: max(target, gmv_vol) or fixed
    **posterior_kwargs
):
    """
    Runs a rolling window backtest for Bayesian strategies that MAY require benchmark data for posterior (e.g., Bayesian CAPM).
    """
    weights_over_time = []
    portfolio_returns_list = []
    portfolio_return_dates = []
    rebalance_dates = []
    T = len(excess_returns_df)
    asset_names_master_list = excess_returns_df.columns.tolist()
    fixed_risk_limit_sq_val = risk_limit_target_vol**2

    for start_idx in range(0, T - estimation_window - holding_period + 1, holding_period):
        est_asset_data = excess_returns_df.iloc[start_idx : start_idx + estimation_window].dropna(axis=1, how='all')
        est_benchmark_data = benchmark_returns_series.iloc[start_idx : start_idx + estimation_window]

        if est_asset_data.empty or est_asset_data.shape[1] == 0:
            print(f"Skipping period at {excess_returns_df.index[start_idx]} due to no valid assets in estimation asset data.")
            continue

        current_asset_names = est_asset_data.columns.tolist()

        hold_asset_data = excess_returns_df.iloc[start_idx + estimation_window : start_idx + estimation_window + holding_period]
        hold_asset_data_aligned = hold_asset_data[current_asset_names]
        current_hold_dates = hold_asset_data_aligned.index

        mu_posterior, Sigma_posterior = posterior_fn(est_asset_data, est_benchmark_data, **posterior_kwargs)

        if Sigma_posterior.shape[0] == 0:
            print(f"Skipping period due to empty Sigma_posterior at {excess_returns_df.index[start_idx + estimation_window]}")
            continue

        current_risk_limit_sq = fixed_risk_limit_sq_val
        if adaptive_risk_target_vol is not None and gmv_vol_calculator_fn is not None:
            gmv_vol = gmv_vol_calculator_fn(mu_posterior, Sigma_posterior)
            if not pd.isna(gmv_vol):
                adaptive_risk_limit_vol = max(adaptive_risk_target_vol, gmv_vol)
                current_risk_limit_sq = adaptive_risk_limit_vol**2
            else:
                print(f"Warning: GMV calculation failed for adaptive risk at {excess_returns_df.index[start_idx + estimation_window]}. Using fixed risk limit.")

        weights = portfolio_solver_fn(mu_posterior, Sigma_posterior, current_risk_limit_sq, constraint_fn, ridge=ridge, lambda_ridge=lambda_ridge)

        if weights is not None:
            port_rets = hold_asset_data_aligned.dot(weights)
            portfolio_returns_list.extend(port_rets)
            portfolio_return_dates.extend(current_hold_dates)
            weights_series = pd.Series(weights, index=current_asset_names)
            weights_over_time.append(weights_series)
            rebalance_dates.append(current_hold_dates[0] if not current_hold_dates.empty else pd.NaT)
        else:
             print(f"Warning: Weights could not be computed for custom Bayesian strategy for period starting {excess_returns_df.index[start_idx + estimation_window]}")

    if not rebalance_dates or pd.isna(rebalance_dates).all():
        print("Warning: No valid rebalance dates for custom Bayesian strategy. Returning empty results.")
        return pd.DataFrame(columns=asset_names_master_list), pd.Series(dtype=float)

    weights_df = pd.DataFrame(weights_over_time, index=rebalance_dates)
    weights_df = weights_df.reindex(columns=asset_names_master_list).fillna(0)

    if not portfolio_returns_list:
        return weights_df, pd.Series(dtype=float, name="Custom Bayesian Portfolio Excess Returns")

    returns_series = pd.Series(portfolio_returns_list, name="Custom Bayesian Portfolio Excess Returns", index=portfolio_return_dates)
    return weights_df, returns_series

def summarize_performance(weights_df, returns_series, risk_free_aligned_for_total_ret=None):
    """
    Summarizes portfolio performance metrics.
    If risk_free_aligned_for_total_ret is provided, calculates total returns and related metrics.
    Otherwise, all metrics are based on excess returns_series.
    """
    # Excess return metrics first
    mean_excess_return = returns_series.mean()
    volatility_excess = returns_series.std()
    sharpe_ratio_excess = (mean_excess_return / volatility_excess * np.sqrt(12)) if volatility_excess != 0 else 0
    annualized_excess_return_percent = ((1 + mean_excess_return) ** 12 - 1) * 100
    cumulative_excess_returns = (1 + returns_series).cumprod()
    total_cumulative_excess_return = cumulative_excess_returns.iloc[-1] - 1 if not cumulative_excess_returns.empty else 0
    
    # If RF provided, calc total return metrics
    if risk_free_aligned_for_total_ret is not None and not returns_series.empty:
        total_returns_series = returns_series + risk_free_aligned_for_total_ret.reindex(returns_series.index, method='ffill')
        mean_total_return = total_returns_series.mean()
        # Volatility usually on excess returns
        # Sharpe: total vs RF (effectively excess return Sharpe)
        annualized_total_return_percent = ((1 + mean_total_return) ** 12 - 1) * 100
        cumulative_total_returns = (1 + total_returns_series).cumprod()
        total_cumulative_total_return = cumulative_total_returns.iloc[-1] - 1 if not cumulative_total_returns.empty else 0
    else: # Default to excess metrics if no RF
        annualized_total_return_percent = annualized_excess_return_percent
        total_cumulative_total_return = total_cumulative_excess_return

    # Rolling HHI (concentration)
    rolling_hhi = weights_df.apply(lambda w: np.sum(w**2), axis=1) if not weights_df.empty else pd.Series(dtype=float)
    avg_hhi = rolling_hhi.mean()
    hhi_std_dev = rolling_hhi.std()

    # Rolling weight std (stability)
    rolling_weight_std = weights_df.std(axis=1) if not weights_df.empty else pd.Series(dtype=float)
    avg_rolling_weight_std = rolling_weight_std.mean()

    # Turnover
    if not weights_df.empty and len(weights_df) > 1:
        turnover_series = weights_df.diff().abs().sum(axis=1).iloc[1:]
        avg_turnover = turnover_series.mean()
        turnover_std_dev = turnover_series.std()
    else:
        avg_turnover = 0
        turnover_std_dev = 0

    summary = pd.DataFrame({
        "Total Cumulative Return (Total)": [total_cumulative_total_return],
        "Mean Monthly Return (Excess)": [mean_excess_return],
        "Volatility (Monthly Std, Excess)": [volatility_excess],
        "Annualized Sharpe Ratio (Excess)": [sharpe_ratio_excess],
        "Annualized Return % (Total)": [annualized_total_return_percent],
        "Average HHI": [avg_hhi],
        "HHI Std Dev": [hhi_std_dev],
        "Avg Rolling Weight Std Dev": [avg_rolling_weight_std],
        "Average Turnover": [avg_turnover],
        "Turnover Std Dev": [turnover_std_dev]
    })

    # Allocation Diagnostics
    if not weights_df.empty:
        high_weight_counts = (weights_df > 0.9).sum()
        weight_std_dev_per_asset = weights_df.std()
        avg_weight_per_asset = weights_df.mean()
    else:
        high_weight_counts = pd.Series(dtype=float)
        weight_std_dev_per_asset = pd.Series(dtype=float)
        avg_weight_per_asset = pd.Series(dtype=float)

    allocation_diagnostics = pd.DataFrame({
        "Average Weight": avg_weight_per_asset,
        "High Weight Count (>90%)": high_weight_counts,
        "Weight Std Dev": weight_std_dev_per_asset
    })

    # Cumul. returns (total or excess)
    final_cumulative_returns_series = (1 + total_returns_series).cumprod() * 100 if risk_free_aligned_for_total_ret is not None and not returns_series.empty else (1 + returns_series).cumprod() * 100

    return summary, allocation_diagnostics, final_cumulative_returns_series, turnover_series if 'turnover_series' in locals() else pd.Series(dtype=float)
