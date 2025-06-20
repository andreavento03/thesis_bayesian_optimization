import pandas as pd
import numpy as np
import scipy.stats as stats # For normality tests

# Import from our utility modules
from utils import data_loader
from utils import portfolio_optimizers
from utils import bayesian_estimators
from utils import backtesting_engine
from utils import plotting_utils

# --- Configuration ---
# Input Data Paths
BASE_DATA_PATH = '../Data/' # Relative to Code/
FINAL_STOCKS_PATH = BASE_DATA_PATH + 'Final Dataset/final_dataset_stocks.csv'
SP500_BENCHMARK_PATH = BASE_DATA_PATH + 'SP_500_benchmark.csv'
RF_MAIN_PATH = BASE_DATA_PATH + '3-month_T_bill.csv' # Main RF data

HISTORICAL_RETURNS_PATH = BASE_DATA_PATH + 'pre-estimation-capm/historical_before2008_returns.csv'
HISTORICAL_RF_PATH = BASE_DATA_PATH + 'DGS1MO-2.csv' # Historical risk-free rate data
HISTORICAL_SP500_BENCHMARK_PATH = BASE_DATA_PATH + 'SP_500_benchmark_presample.csv'

# Output Paths
PLOTS_DIR = 'plots_output/'
RESULTS_DIR = 'results_output/'

# General Parameters
MAIN_RF_START_DATE = "2008-01-31"
MAIN_RF_END_DATE = "2022-12-31"
HISTORICAL_RF_START_DATE = "2002-01-31" # For pre-sample/prior data
HISTORICAL_RF_END_DATE = "2007-12-31"

# Asset Ticker Mapping (from original notebook)
NAME_TO_TICKER_MAP = {
    'A T & T INC': 'T', 'ALPHABET INC': 'GOOGL', 'ALTRIA GROUP INC': 'MO',
    'AMERICAN INTERNATIONAL GROUP INC': 'AIG', 'APPLE INC': 'AAPL', 'BANK OF AMERICA CORP': 'BAC',
    'BERKSHIRE HATHAWAY INC DEL': 'BRK.B', 'CHEVRON CORP NEW': 'CVX', 'CISCO SYSTEMS INC': 'CSCO',
    'CITIGROUP INC': 'C', 'COCA COLA CO': 'KO', 'EXXON MOBIL CORP': 'XOM',
    'GENERAL ELECTRIC CO': 'GE', 'INTEL CORP': 'INTC', 'INTERNATIONAL BUSINESS MACHS COR': 'IBM',
    'JOHNSON & JOHNSON': 'JNJ', 'MICROSOFT CORP': 'MSFT', 'PFIZER INC': 'PFE',
    'PROCTER & GAMBLE CO': 'PG', 'WALMART INC': 'WMT', 'b20ret': 'BOND20'
}

# Backtesting Parameters
ESTIMATION_WINDOW = 60
HOLDING_PERIOD = 2
RISK_LIMIT_TARGET_VOL = 0.03 # Monthly target volatility

# --- Helper Functions ---
def run_normality_tests(df, label=''):
    """Performs Shapiro-Wilk, Jarque-Bera, and Anderson-Darling tests for normality."""
    results = []
    for col in df.columns:
        data = df[col].dropna()
        if len(data) < 3: # Shapiro: min 3 samples
            print(f"Skipping normality tests for {col} due to insufficient data ({len(data)} samples).")
            continue

        shapiro_stat, shapiro_p = stats.shapiro(data)
        jb_stat, jb_p = stats.jarque_bera(data)
        ad_result = stats.anderson(data, dist='norm')
        ad_stat = ad_result.statistic
        ad_crit_val_5_percent = ad_result.critical_values[2] # 5% significance level

        results.append({
            'Asset': col,
            'Shapiro_p': shapiro_p,
            'JarqueBera_p': jb_p,
            'AD_stat': ad_stat,
            'AD_5%_crit': ad_crit_val_5_percent,
            'AD_pass_5%': ad_stat < ad_crit_val_5_percent
        })

    test_results_df = pd.DataFrame(results)
    if not test_results_df.empty:
        test_results_df['Shapiro_pass_5%'] = test_results_df['Shapiro_p'] > 0.05
        test_results_df['JB_pass_5%'] = test_results_df['JarqueBera_p'] > 0.05
        all_passed_col_name = 'All_Passed_5%'
        all_failed_col_name = 'All_Failed_5%'
        test_results_df[all_passed_col_name] = test_results_df['Shapiro_pass_5%'] & \
                                               test_results_df['JB_pass_5%'] & \
                                               test_results_df['AD_pass_5%']
        test_results_df[all_failed_col_name] = ~test_results_df['Shapiro_pass_5%'] & \
                                               ~test_results_df['JB_pass_5%'] & \
                                               ~test_results_df['AD_pass_5%']

        summary_counts = {
            'Total Assets Tested': len(test_results_df),
            'Shapiro Passed (5%)': test_results_df['Shapiro_pass_5%'].sum(),
            'Jarque-Bera Passed (5%)': test_results_df['JB_pass_5%'].sum(),
            'Anderson-Darling Passed (5%)': test_results_df['AD_pass_5%'].sum(),
            'Passed All Tests (5%)': test_results_df[all_passed_col_name].sum(),
            'Failed All Tests (5%)': test_results_df[all_failed_col_name].sum()
        }
        summary_counts['Partial Agreement (5%)'] = summary_counts['Total Assets Tested'] - \
                                                 summary_counts['Passed All Tests (5%)'] - \
                                                 summary_counts['Failed All Tests (5%)']
        summary_df = pd.DataFrame.from_dict(summary_counts, orient='index', columns=[f'{label} Normality Test Counts'])
    else:
        summary_df = pd.DataFrame(columns=[f'{label} Normality Test Counts'])

    return test_results_df, summary_df

# --- Main Analysis Workflow ---
def main():
    """Main function to run the MVO analysis workflow."""
    print("Starting MVO Analysis Workflow...")

    # Create output dirs
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Load Core Data ---
    print("\n--- Section 1: Loading Core Data ---")
    core_returns_df = data_loader.load_core_asset_returns(
        FINAL_STOCKS_PATH, FINAL_BONDS_PATH, NAME_TO_TICKER_MAP
    )
    print("Core asset returns loaded:")
    print(core_returns_df.head())

    benchmark_returns_series = data_loader.load_benchmark_returns(
        SP500_BENCHMARK_PATH, core_returns_df.index
    )
    print("\nBenchmark returns loaded and aligned.")

    risk_free_series = data_loader.load_risk_free_rates(
        RF_MAIN_PATH, MAIN_RF_START_DATE, MAIN_RF_END_DATE, core_returns_df.index
    )
    print("\nRisk-free rates loaded and aligned.")

    excess_returns_df = data_loader.calculate_excess_returns(core_returns_df, risk_free_series)
    print("\nExcess returns calculated:")
    print(excess_returns_df.head())

    benchmark_excess_returns_series = data_loader.calculate_excess_returns(
        benchmark_returns_series.to_frame(name='sp500'), # Convert to DataFrame for subtraction
        risk_free_series
    )['sp500'] # Convert back to Series
    print("\nBenchmark excess returns calculated.")

    # --- 2. Initial EDA & Benchmark Comparison ---
    print("\n--- Section 2: Initial EDA & Benchmark Comparison ---")

    # Cumulative Wealth Plot (from original cell In[19])
    # Prep wealth data ($100 init.)
    wealth_df = (1 + core_returns_df).cumprod() * 100
    benchmark_wealth_series = (1 + benchmark_returns_series).cumprod() * 100
    benchmark_wealth_series.name = "S&P 500 Benchmark" # Set name for legend

    # Identify top/bottom performers for highlighting (example from original notebook)
    if not wealth_df.empty:
        final_wealth = wealth_df.iloc[-1].sort_values(ascending=False)
        highlighted_assets_list = []
        if len(final_wealth) >= 1: highlighted_assets_list.append(final_wealth.index[0])
        if len(final_wealth) >= 2: highlighted_assets_list.append(final_wealth.index[1])
        if len(final_wealth) >= 3: highlighted_assets_list.append(final_wealth.index[2])
        if len(final_wealth) >= 4: highlighted_assets_list.append(final_wealth.index[-1]) # worst
        if len(final_wealth) >= 5: highlighted_assets_list.append(final_wealth.index[-2]) # second worst
    else:
        highlighted_assets_list = None

    plotting_utils.plot_cumulative_wealth(
        wealth_df,
        benchmark_series=benchmark_wealth_series,
        title="Cumulative Wealth per Company vs. S&P 500 Benchmark",
        highlight_assets=highlighted_assets_list,
        output_path=os.path.join(PLOTS_DIR, "cumulative_wealth_vs_benchmark.png")
    )
    print("Cumulative wealth plot generated.")

    # Annualized Returns Plot (from original cell In[21])
    mean_monthly_core_returns = core_returns_df.mean()
    mean_monthly_benchmark_return = benchmark_returns_series.mean()
    annualized_core_returns = ((1 + mean_monthly_core_returns) ** 12 - 1) * 100
    annualized_benchmark_return = ((1 + mean_monthly_benchmark_return) ** 12 - 1) * 100

    plot_ann_returns_data = annualized_core_returns.copy()
    plot_ann_returns_data['S&P 500'] = annualized_benchmark_return # Add benchmark for plotting
    plot_ann_returns_data = plot_ann_returns_data.sort_values(ascending=False)

    plotting_utils.plot_annualized_returns_bar(
        plot_ann_returns_data,
        benchmark_name="S&P 500",
        title="Annualized Return: Assets vs. S&P 500 Benchmark",
        output_path=os.path.join(PLOTS_DIR, "annualized_returns_bar.png")
    )
    print("Annualized returns bar chart generated.")

    # --- 3. Load Historical Data for Priors ---
    print("\n--- Section 3: Loading Historical Data for Priors ---")
    # For Bayesian priors

    # Target cols: current core assets
    historical_asset_returns_prior_df = data_loader.load_historical_asset_returns_for_prior(
        HISTORICAL_RETURNS_PATH,
        target_columns=core_returns_df.columns # Use columns from the main analysis period
    )
    print("Historical asset returns for priors loaded.")

    if not historical_asset_returns_prior_df.empty:
        historical_rf_prior_series = data_loader.load_historical_risk_free_for_prior(
            HISTORICAL_RF_PATH, HISTORICAL_RF_START_DATE, HISTORICAL_RF_END_DATE,
            historical_asset_returns_prior_df.index
        )
        print("Historical risk-free rates for priors loaded.")

        historical_excess_returns_prior_df = data_loader.calculate_excess_returns(
            historical_asset_returns_prior_df, historical_rf_prior_series
        )
        print("Historical excess returns for priors calculated.")
        historical_excess_returns_prior_df.to_csv(os.path.join(RESULTS_DIR, "historical_excess_returns_for_priors.csv"))

        # Sample hist cov (for Bayesian prior)
        # This is used as a potential source for Omega in informative_bayes_predictive_niw
        sample_historical_covariance_prior = historical_excess_returns_prior_df.cov()
        sample_historical_covariance_prior.to_csv(os.path.join(RESULTS_DIR, "sample_historical_covariance_for_priors.csv"))
        print("Sample historical covariance for priors calculated and saved.")

    else:
        print("Skipping historical prior data processing as historical asset returns are empty.")
        historical_excess_returns_prior_df = pd.DataFrame() # Ensure it's defined
        sample_historical_covariance_prior = pd.DataFrame()

    # --- 4. Normality Analysis ---
    print("\n--- Section 4: Normality Analysis ---")
    # Log returns for normality tests
    log_core_returns_df = np.log(1 + core_returns_df).dropna(how='all')

    raw_normality_results_df, raw_normality_summary_df = run_normality_tests(core_returns_df, "Raw Core Returns")
    log_normality_results_df, log_normality_summary_df = run_normality_tests(log_core_returns_df, "Log Core Returns")

    print("\nRaw Returns Normality Summary:")
    print(raw_normality_summary_df)
    print("\nLog Returns Normality Summary:")
    print(log_normality_summary_df)

    raw_normality_results_df.to_csv(os.path.join(RESULTS_DIR, "normality_tests_raw_returns.csv"))
    log_normality_results_df.to_csv(os.path.join(RESULTS_DIR, "normality_tests_log_returns.csv"))
    print("Normality test detailed results saved to CSV.")

    plotting_utils.plot_return_distribution_histograms(
        core_returns_df,
        log_returns_data_df=log_core_returns_df,
        title_prefix="Asset Return Distributions",
        output_path_raw=os.path.join(PLOTS_DIR, "return_histograms_raw.png"),
        output_path_log=os.path.join(PLOTS_DIR, "return_histograms_log.png"),
        output_path_sampled_comparison=os.path.join(PLOTS_DIR, "return_histograms_sampled_comparison.png")
    )
    print("Return distribution histograms generated.")

    # --- 5. Sensitivity to Mean Estimate ---
    print("\n--- Section 5: Sensitivity to Mean Estimate ---")
    mu_hat_excess = excess_returns_df.mean().values
    sigma_hat_excess = excess_returns_df.cov().values
    asset_names_excess = excess_returns_df.columns

    # Baseline MVO (unconstrained)
    baseline_weights = portfolio_optimizers.compute_mvo_weights(mu_hat_excess, sigma_hat_excess)

    # Perturb mean (Gaussian noise as in original notebook In[50])
    np.random.seed(123) # for reproducibility
    epsilon_mu = np.random.normal(loc=0, scale=0.001, size=mu_hat_excess.shape)
    mu_noisy = mu_hat_excess + epsilon_mu

    weights_noisy_mean = portfolio_optimizers.compute_mvo_weights(mu_noisy, sigma_hat_excess)

    weights_comparison_dict = {
        'Baseline': pd.Series(baseline_weights, index=asset_names_excess),
        'Perturbed Mean (N(0, 0.001))': pd.Series(weights_noisy_mean, index=asset_names_excess)
    }
    plotting_utils.plot_weights_comparison_bar(
        weights_comparison_dict,
        title="Portfolio Weights: Baseline vs. Perturbed Mean",
        output_path=os.path.join(PLOTS_DIR, "weights_sensitivity_mean_perturbation.png")
    )
    print("Mean sensitivity plot (perturbation) generated.")

    # Sensitivity to estimation window (original cell In[54])
    window_sizes = [40, 60, 80] # Months
    weights_by_window_dict = {}
    # Use overall mean for mu_win, but vary covariance matrix by window
    # The original notebook did: mu_win = excess_returns.mean().values (fixed for all windows)
    # And Sigma_win = recent_returns.cov().values (varied) - this is a bit unusual.
    # Typically both mu and Sigma would be from the window.
    # For this refactoring, I will follow the original notebook's logic.
    mu_for_window_sensitivity = excess_returns_df.mean().values

    for window in window_sizes:
        if len(excess_returns_df) >= window:
            recent_returns = excess_returns_df.iloc[-window:]
            sigma_win = recent_returns.cov().values
            # Ensure mu_for_window_sensitivity matches columns of sigma_win if assets changed due to NaNs
            # This requires aligning mu_for_window_sensitivity to assets in recent_returns.columns
            # For simplicity here, assume asset set is stable or handle alignment if needed.
            # If assets can change, mu should be recalculated for `recent_returns.columns`
            # mu_win_aligned = excess_returns_df[recent_returns.columns].mean().values

            # Sticking to original: use fixed mu_for_window_sensitivity, ensure it's aligned if assets differ
            # This is tricky if asset universe changes. A safer way:
            current_assets = recent_returns.columns
            mu_win_aligned = excess_returns_df[current_assets].mean().values
            sigma_win_aligned = excess_returns_df[current_assets].iloc[-window:].cov().values

            if mu_win_aligned.shape[0] == sigma_win_aligned.shape[0]: # Check for consistent dimensions
                 weights_by_window_dict[f"{window} months"] = pd.Series(
                     portfolio_optimizers.compute_mvo_weights(mu_win_aligned, sigma_win_aligned),
                     index=current_assets
                 )
            else:
                print(f"Skipping window {window} due to dimension mismatch after asset selection.")
        else:
            print(f"Skipping window {window} as it's larger than available data {len(excess_returns_df)}.")

    if weights_by_window_dict: # Check if dict is not empty
        # Need to make sure all series in dict have the same full index for plotting
        # Reindex each series to include all assets, filling missing with 0
        all_assets_in_windows = pd.Index([])
        for s in weights_by_window_dict.values():
            all_assets_in_windows = all_assets_in_windows.union(s.index)

        aligned_weights_by_window_dict = {
            key: s.reindex(all_assets_in_windows).fillna(0) for key, s in weights_by_window_dict.items()
        }

        plotting_utils.plot_weights_comparison_bar(
            aligned_weights_by_window_dict,
            title="MVO Weights Across Different Estimation Window Lengths",
            output_path=os.path.join(PLOTS_DIR, "weights_sensitivity_estimation_window.png")
        )
        print("Estimation window sensitivity plot generated.")
    else:
        print("No data for estimation window sensitivity plot.")

    # --- 6. Classical Efficient Frontier ---
    print("\n--- Section 6: Classical Efficient Frontier ---")
    # Using excess returns for the main analysis period (all available data)
    # The bayesian_scale_factor_fn is None for classical
    target_returns_classical, portfolio_vols_classical, weights_classical_on_frontier = \
        portfolio_optimizers.compute_efficient_frontier_general(excess_returns_df, bayesian_scale_factor_fn=None)

    # Prepare data for plotting_utils.plot_efficient_frontier
    # It expects a list of tuples: (vols, rets, label, style_dict)
    # Or (vols, rets, label, gmv_point, tan_point, style_dict)

    # Calculate GMV and Tangency points for this frontier
    rf_for_tangency = 0.0 # Assuming risk-free rate for Sharpe is 0 for excess returns

    # Filter NaNs from optimization
    valid_indices_classical = ~np.isnan(portfolio_vols_classical) & ~np.isnan(target_returns_classical)
    portfolio_vols_classical_valid = portfolio_vols_classical[valid_indices_classical]
    target_returns_classical_valid = target_returns_classical[valid_indices_classical]

    if len(portfolio_vols_classical_valid) > 0:
        gmv_idx_classical = np.nanargmin(portfolio_vols_classical_valid)
        gmv_point_classical = (portfolio_vols_classical_valid[gmv_idx_classical], target_returns_classical_valid[gmv_idx_classical])

        sharpe_ratios_classical = (target_returns_classical_valid - rf_for_tangency) / portfolio_vols_classical_valid
        tangency_idx_classical = np.nanargmax(sharpe_ratios_classical)
        tangency_point_classical = (portfolio_vols_classical_valid[tangency_idx_classical], target_returns_classical_valid[tangency_idx_classical])

        classical_frontier_plot_data = [(
            portfolio_vols_classical_valid,
            target_returns_classical_valid,
            "Classical MVO",
            gmv_point_classical,
            tangency_point_classical,
            {'line_color': '#0d47a1', 'dot_color': '#1976d2', 'line_style': '-'}
        )]

        plotting_utils.plot_efficient_frontier(
            classical_frontier_plot_data,
            title="Classical Efficient Frontier (Excess Returns)",
            plot_cml=True, # Plot Capital Market Line based on this frontier
            risk_free_rate=rf_for_tangency,
            output_path=os.path.join(PLOTS_DIR, "efficient_frontier_classical.png")
        )
        print("Classical efficient frontier plot generated.")
    else:
        print("Could not generate classical efficient frontier; not enough valid points.")

    # Original notebook also plotted frontiers for different estimation windows (cell In[55])
    # This can be done by looping through window_sizes again if desired,
    # calling compute_efficient_frontier_general for each subset of data.
    # Example:
    frontiers_by_window_plot_data = []
    for w_size in window_sizes:
        if len(excess_returns_df) >= w_size:
            windowed_returns = excess_returns_df.iloc[:w_size] # Using start of data for consistency
            if windowed_returns.shape[1] < 2: # Need at least 2 assets
                print(f"Skipping frontier for window {w_size}, not enough assets: {windowed_returns.shape[1]}")
                continue

            tr_w, pv_w, _ = portfolio_optimizers.compute_efficient_frontier_general(windowed_returns)

            valid_indices_w = ~np.isnan(pv_w) & ~np.isnan(tr_w)
            if np.sum(valid_indices_w) > 1: # Need at least 2 points to plot a line
                 frontiers_by_window_plot_data.append(
                     (pv_w[valid_indices_w], tr_w[valid_indices_w], f"{w_size} months",
                      {'line_style': '--'}) # Simple style, no GMV/Tan for these
                 )
            else:
                print(f"Not enough valid points to plot frontier for window {w_size}.")

    if frontiers_by_window_plot_data:
        plotting_utils.plot_efficient_frontier(
            frontiers_by_window_plot_data,
            title="Efficient Frontiers for Different Estimation Windows",
            output_path=os.path.join(PLOTS_DIR, "efficient_frontiers_by_window.png")
        )
        print("Efficient frontiers by window plot generated.")
    else:
        print("No data for efficient frontiers by window plot.")

    # --- 7. Rolling Window Backtesting (Classical Strategies) ---
    print("\n--- Section 7: Rolling Window Backtesting (Classical Strategies) ---")

    # Define strategies from original notebook (cell In[61])
    # Constraint functions are in portfolio_optimizers
    classical_strategies_to_run = {
        "Unconstrained": portfolio_optimizers.unconstrained,
        "Unconstrained Ridge": portfolio_optimizers.unconstrained_ridge, # solve_portfolio handles ridge flag
        "Long Only": portfolio_optimizers.long_only,
        # "Cap 0.5": portfolio_optimizers.capped_0_5, # Not used in final plots of notebook
        # "Cap 0.3": portfolio_optimizers.capped_0_3  # Not used in final plots of notebook
    }

    classical_results_collection = {}
    classical_cumulative_returns_collection = {}
    classical_weights_collection = {}

    # Ensure risk_free_series is available for total return calculation in summarize_performance
    # It was loaded as 'risk_free_series' earlier.

    for strat_name, constraint_func in classical_strategies_to_run.items():
        print(f"\nRunning classical strategy: {strat_name}...")

        use_ridge = "Ridge" in strat_name

        weights_df, strategy_excess_returns_series = backtesting_engine.run_rolling_strategy(
            excess_returns_df=excess_returns_df,
            estimation_window=ESTIMATION_WINDOW,
            holding_period=HOLDING_PERIOD,
            risk_limit_target_vol=RISK_LIMIT_TARGET_VOL,
            constraint_fn=constraint_func,
            portfolio_solver_fn=portfolio_optimizers.solve_portfolio, # Pass the actual solver
            ridge=use_ridge,
            lambda_ridge=0.1 # Default from notebook, make configurable if needed
        )

        summary_df, allocation_diag_df, cum_total_returns_rebased100, _ = \
            backtesting_engine.summarize_performance(
                weights_df,
                strategy_excess_returns_series,
                risk_free_aligned_for_total_ret=risk_free_series # Pass aligned rf series
            )

        classical_results_collection[strat_name] = summary_df
        classical_cumulative_returns_collection[strat_name] = cum_total_returns_rebased100
        classical_weights_collection[strat_name] = weights_df

        print(f"Summary for {strat_name}:")
        print(summary_df)
        summary_df.to_csv(os.path.join(RESULTS_DIR, f"summary_classical_{strat_name.lower().replace(' ', '_')}.csv"))
        weights_df.to_csv(os.path.join(RESULTS_DIR, f"weights_classical_{strat_name.lower().replace(' ', '_')}.csv"))

        # Plot average weights for this strategy
        avg_weights_this_strategy = allocation_diag_df["Average Weight"]
        plotting_utils.plot_average_weights_bar(
            avg_weights_this_strategy,
            strategy_name=strat_name,
            output_path=os.path.join(PLOTS_DIR, f"avg_weights_classical_{strat_name.lower().replace(' ', '_')}.png")
        )

    # Plot comparison of cumulative returns for all classical strategies
    # Prep benchmark for comparison
    if classical_cumulative_returns_collection:
        # Align benchmark to the start date of the strategy returns
        # Find the earliest start date among all strategy returns
        min_date = min(s.index.min() for s in classical_cumulative_returns_collection.values() if not s.empty)

        # Filter benchmark returns from this min_date onwards
        benchmark_returns_for_plot = benchmark_returns_series[benchmark_returns_series.index >= min_date]
        # Rebase to 100 starting from this common min_date
        benchmark_cum_total_for_plot = (1 + benchmark_returns_for_plot).cumprod()
        if not benchmark_cum_total_for_plot.empty:
             benchmark_cum_total_for_plot = (benchmark_cum_total_for_plot / benchmark_cum_total_for_plot.iloc[0]) * 100
             benchmark_cum_total_for_plot.name = "S&P 500"
        else:
            benchmark_cum_total_for_plot = None # In case of no overlapping benchmark data

        plotting_utils.plot_strategy_cumulative_returns_comparison(
            classical_cumulative_returns_collection,
            benchmark_series=benchmark_cum_total_for_plot, # Pass the rebased and aligned benchmark
            benchmark_label="S&P 500",
            title="Cumulative Value: Classical Strategies vs Benchmark",
            output_path=os.path.join(PLOTS_DIR, "cumulative_returns_classical_strategies.png")
        )
        print("Classical strategies cumulative returns comparison plot generated.")

        # Consolidate all classical summaries for metrics comparison plot
        all_classical_summaries_df = pd.concat(
            [df.assign(Strategy=name) for name, df in classical_results_collection.items()]
        ).set_index("Strategy")

        # Add S&P 500 benchmark metrics for comparison (from original cell In[27], but using excess returns)
        # Align benchmark stats to backtest period
        # For simplicity, we'll use the full-period benchmark excess returns stats here.
        # A more rigorous approach would align the benchmark stats calculation to the actual backtest period.

        # Calculate S&P 500 metrics over the common backtest period if possible
        if benchmark_cum_total_for_plot is not None and not benchmark_excess_returns_series.empty:
            common_idx = benchmark_excess_returns_series.index.intersection(min_date if min_date else benchmark_excess_returns_series.index.min(),
                                                                         max(s.index.max() for s in classical_cumulative_returns_collection.values() if not s.empty))

            if not common_idx.empty:
                aligned_bench_excess_for_stats = benchmark_excess_returns_series.loc[common_idx]

                bench_mean_monthly_exc = aligned_bench_excess_for_stats.mean()
                bench_vol_monthly_exc = aligned_bench_excess_for_stats.std()
                bench_sharpe_exc = (bench_mean_monthly_exc / bench_vol_monthly_exc * np.sqrt(12)) if bench_vol_monthly_exc != 0 else 0
                bench_ann_ret_exc_pct = ((1 + bench_mean_monthly_exc)**12 - 1) * 100
                # For total cumulative return of benchmark, we use the rebased one for the plot period
                # This is a bit of a mix, but aligns with how plots are often presented.
                # summarize_performance returns 'Total Cumulative Return (Total)', so we need total for benchmark too.
                aligned_bench_total_for_stats = benchmark_returns_series.loc[common_idx]
                bench_total_cum_ret = (1 + aligned_bench_total_for_stats).prod() -1


                benchmark_summary_for_plot = pd.DataFrame({
                    "Total Cumulative Return (Total)": [bench_total_cum_ret],
                    "Mean Monthly Return (Excess)": [bench_mean_monthly_exc],
                    "Volatility (Monthly Std, Excess)": [bench_vol_monthly_exc],
                    "Annualized Sharpe Ratio (Excess)": [bench_sharpe_exc],
                    "Annualized Return % (Total)": [((1+aligned_bench_total_for_stats.mean())**12-1)*100], # Ann Total Ret
                    "Average HHI": [np.nan], # Not applicable for benchmark
                    "HHI Std Dev": [np.nan],
                    "Avg Rolling Weight Std Dev": [np.nan],
                    "Average Turnover": [0], # No turnover for buy-and-hold benchmark
                    "Turnover Std Dev": [0]
                }, index=["S&P 500"])
                all_classical_summaries_df = pd.concat([all_classical_summaries_df, benchmark_summary_for_plot])

        metrics_to_plot_classical = [
            "Total Cumulative Return (Total)", "Mean Monthly Return (Excess)",
            "Volatility (Monthly Std, Excess)", "Annualized Sharpe Ratio (Excess)",
            "Annualized Return % (Total)", "Average HHI", "Average Turnover"
        ]
        maximize_metrics_classical = {
            "Total Cumulative Return (Total)", "Mean Monthly Return (Excess)",
            "Annualized Sharpe Ratio (Excess)", "Annualized Return % (Total)"
        }

        plotting_utils.plot_strategy_metrics_comparison_bar(
            all_classical_summaries_df,
            metrics_to_plot=metrics_to_plot_classical,
            benchmark_name="S&P 500",
            maximize_metrics=maximize_metrics_classical,
            output_path=os.path.join(PLOTS_DIR, "metrics_comparison_classical_strategies.png")
        )
        print("Classical strategies metrics comparison plot generated.")

    # --- 8. Bayesian Efficient Frontiers (Diffuse Prior & Conjugate NIW) ---
    print("\n--- Section 8: Bayesian Efficient Frontiers ---")

    # 8.1 Diffuse Prior (Jeffreys') Efficient Frontier (Original Cell In[69])
    # Define diffuse prior scaling fn
    def diffuse_prior_scale_factor(T, N):
        if T <= N + 2: return 1.0 # Fallback to no scaling if condition not met
        return (1 + 1/T) * ((T - 1) / (T - N - 2))

    target_returns_bayes_diffuse, portfolio_vols_bayes_diffuse, _ = \
        portfolio_optimizers.compute_efficient_frontier_general(
            excess_returns_df,
            bayesian_scale_factor_fn=diffuse_prior_scale_factor
        )

    frontiers_to_plot = []
    # Add Classical frontier data (calculated in Section 6) for comparison
    if 'portfolio_vols_classical_valid' in locals() and len(portfolio_vols_classical_valid) > 0:
        frontiers_to_plot.append(
            (portfolio_vols_classical_valid, target_returns_classical_valid, "Classical MVO",
             gmv_point_classical, tangency_point_classical, # from Sec 6
             {'line_color': '#0d47a1', 'dot_color': '#1976d2', 'line_style': '-'})
        )

    # Add Bayesian Diffuse frontier
    valid_bayes_diffuse = ~np.isnan(portfolio_vols_bayes_diffuse) & ~np.isnan(target_returns_bayes_diffuse)
    if np.sum(valid_bayes_diffuse) > 1:
        # Calculate GMV/Tangency for diffuse
        gmv_idx_bd = np.nanargmin(portfolio_vols_bayes_diffuse[valid_bayes_diffuse])
        gmv_pt_bd = (portfolio_vols_bayes_diffuse[valid_bayes_diffuse][gmv_idx_bd], target_returns_bayes_diffuse[valid_bayes_diffuse][gmv_idx_bd])

        sharpe_bd = (target_returns_bayes_diffuse[valid_bayes_diffuse] - rf_for_tangency) / portfolio_vols_bayes_diffuse[valid_bayes_diffuse]
        tan_idx_bd = np.nanargmax(sharpe_bd)
        tan_pt_bd = (portfolio_vols_bayes_diffuse[valid_bayes_diffuse][tan_idx_bd], target_returns_bayes_diffuse[valid_bayes_diffuse][tan_idx_bd])

        frontiers_to_plot.append(
            (portfolio_vols_bayes_diffuse[valid_bayes_diffuse],
             target_returns_bayes_diffuse[valid_bayes_diffuse],
             "Bayesian MVO (Diffuse Prior)",
             gmv_pt_bd, tan_pt_bd,
             {'line_color': '#e64a19', 'dot_color': '#ff7043', 'line_style': '--'})
        )
    else:
        print("Not enough valid points for Bayesian Diffuse frontier.")

    # 8.2 Conjugate NIW Priors Efficient Frontiers (Original Cells In[70]-In[74])

    # Non-informative Conjugate NIW (Original Cell In[72]/[73])
    # Use NIW with flat/default priors
    mu_niw_noninf, sigma_niw_noninf = bayesian_estimators.informative_bayes_predictive_niw(
        excess_returns_df,
        use_flat_mean_prior=True, # Default eta = 0
        tau_prior_confidence=1e-6, # Very low confidence in flat prior mean (original notebook used tau=15 with specific eta)
                                 # For a truly "non-informative" mean, tau should be near zero.
        use_flat_cov_prior=True   # Default Omega = scaled identity
    )
    tr_niw_noninf, pv_niw_noninf, _ = portfolio_optimizers.compute_efficient_frontier_from_moments(
        mu_niw_noninf, sigma_niw_noninf
    )
    valid_niw_noninf = ~np.isnan(pv_niw_noninf) & ~np.isnan(tr_niw_noninf)
    if np.sum(valid_niw_noninf) > 1:
        gmv_idx_niw_noninf = np.nanargmin(pv_niw_noninf[valid_niw_noninf])
        gmv_pt_niw_noninf = (pv_niw_noninf[valid_niw_noninf][gmv_idx_niw_noninf], tr_niw_noninf[valid_niw_noninf][gmv_idx_niw_noninf])
        sharpe_niw_noninf = (tr_niw_noninf[valid_niw_noninf] - rf_for_tangency) / pv_niw_noninf[valid_niw_noninf]
        tan_idx_niw_noninf = np.nanargmax(sharpe_niw_noninf)
        tan_pt_niw_noninf = (pv_niw_noninf[valid_niw_noninf][tan_idx_niw_noninf], tr_niw_noninf[valid_niw_noninf][tan_idx_niw_noninf])

        frontiers_to_plot.append(
            (pv_niw_noninf[valid_niw_noninf],
             tr_niw_noninf[valid_niw_noninf],
             "Bayesian (Conjugate Non-Informative)",
             gmv_pt_niw_noninf, tan_pt_niw_noninf,
             {'line_color': '#08950F', 'dot_color': '#08950F', 'line_style': ':'})
        )
    else:
        print("Not enough valid points for Bayesian Conjugate Non-Informative frontier.")

    # History-informed Conjugate NIW (Original Cell In[72])
    # Requires hist. data from Sec 3
    if not historical_excess_returns_prior_df.empty and not sample_historical_covariance_prior.empty:
        T_hist, N_hist = historical_excess_returns_prior_df.shape

        # Align historical data to current asset universe (excess_returns_df.columns)
        # This is important if assets differ between historical and current periods.
        common_assets_hist = excess_returns_df.columns.intersection(historical_excess_returns_prior_df.columns)

        if len(common_assets_hist) > 0:
            aligned_hist_excess_returns = historical_excess_returns_prior_df[common_assets_hist]

            # Prior mean: mean of historical excess returns for common assets
            eta_hist_prior = aligned_hist_excess_returns.mean().values

            # Prior scale matrix Omega: from sample_historical_covariance_prior, aligned
            # The scaling (nu-N-1) is tricky. Original notebook used: historical_var * 1e-6 * np.eye(N) OR sample_historical_covariance * (nu - N - 1)
            # Let's use sample_historical_covariance_prior for Omega, aligned.
            # nu_prior_dof_hist should be > N_common_assets + 1 for Omega to be scaled properly.
            N_common_assets = len(common_assets_hist)
            nu_prior_dof_hist = float(N_common_assets + 2) # Minimal proper

            # Align sample_historical_covariance_prior to common_assets_hist
            aligned_sample_hist_cov_prior = sample_historical_covariance_prior.loc[common_assets_hist, common_assets_hist]

            omega_hist_prior = aligned_sample_hist_cov_prior * (nu_prior_dof_hist - N_common_assets - 1)

            # We need to call informative_bayes_predictive_niw with excess_returns_df[common_assets_hist]
            # as the main data, and priors derived from aligned_hist_excess_returns.
            mu_niw_hist, sigma_niw_hist = bayesian_estimators.informative_bayes_predictive_niw(
                excess_returns_df[common_assets_hist], # Use current period data for common assets
                eta_prior_mean=eta_hist_prior,
                tau_prior_confidence=15, # From original notebook's example
                nu_prior_dof=nu_prior_dof_hist,
                omega_prior_scale_matrix=omega_hist_prior,
                use_flat_mean_prior=False, # We are providing eta
                use_flat_cov_prior=False  # We are providing omega
            )
            tr_niw_hist, pv_niw_hist, _ = portfolio_optimizers.compute_efficient_frontier_from_moments(
                mu_niw_hist, sigma_niw_hist
            )
            valid_niw_hist = ~np.isnan(pv_niw_hist) & ~np.isnan(tr_niw_hist)
            if np.sum(valid_niw_hist) > 1:
                gmv_idx_niw_hist = np.nanargmin(pv_niw_hist[valid_niw_hist])
                gmv_pt_niw_hist = (pv_niw_hist[valid_niw_hist][gmv_idx_niw_hist], tr_niw_hist[valid_niw_hist][gmv_idx_niw_hist])
                sharpe_niw_hist = (tr_niw_hist[valid_niw_hist] - rf_for_tangency) / pv_niw_hist[valid_niw_hist]
                tan_idx_niw_hist = np.nanargmax(sharpe_niw_hist)
                tan_pt_niw_hist = (pv_niw_hist[valid_niw_hist][tan_idx_niw_hist], tr_niw_hist[valid_niw_hist][tan_idx_niw_hist])

                frontiers_to_plot.append(
                    (pv_niw_hist[valid_niw_hist],
                     tr_niw_hist[valid_niw_hist],
                     "Bayesian (History-Informed Conjugate)",
                     gmv_pt_niw_hist, tan_pt_niw_hist,
                     {'line_color': '#FFC107', 'dot_color': '#FFA000', 'line_style': '-.'})
                )
            else:
                print("Not enough valid points for Bayesian History-Informed Conjugate frontier.")
        else:
            print("No common assets between historical and current period for History-Informed prior.")


    if frontiers_to_plot:
        # Common x/y limits for plot
        min_vol_all = min(np.nanmin(f[0]) for f in frontiers_to_plot if len(f[0]) > 0 and np.sum(~np.isnan(f[0])) > 0)
        max_vol_all = max(np.nanmax(f[0]) for f in frontiers_to_plot if len(f[0]) > 0 and np.sum(~np.isnan(f[0])) > 0)
        min_ret_all = min(np.nanmin(f[1]) for f in frontiers_to_plot if len(f[1]) > 0 and np.sum(~np.isnan(f[1])) > 0)
        max_ret_all = max(np.nanmax(f[1]) for f in frontiers_to_plot if len(f[1]) > 0 and np.sum(~np.isnan(f[1])) > 0)

        # Add some padding
        xlim_eff = (min_vol_all * 0.9, max_vol_all * 1.1) if not (np.isnan(min_vol_all) or np.isinf(min_vol_all)) else None
        ylim_eff = (min_ret_all * 0.9 if min_ret_all > 0 else min_ret_all * 1.1, max_ret_all * 1.1) if not (np.isnan(min_ret_all) or np.isinf(min_ret_all)) else None


        plotting_utils.plot_efficient_frontier(
            frontiers_to_plot,
            title="Efficient Frontier Comparison: Classical vs. Bayesian (Various Priors)",
            xlim=xlim_eff, # Use common limits for x
            ylim=ylim_eff, # Use common limits for y
            output_path=os.path.join(PLOTS_DIR, "efficient_frontiers_bayesian_comparison.png")
        )
        print("Bayesian efficient frontiers comparison plot generated.")
    else:
        print("No frontiers to plot for Bayesian comparison.")

    # --- 9. Bayesian Rolling Window Strategies (Non-CAPM based) ---
    print("\n--- Section 9: Bayesian Rolling Window Strategies (Non-CAPM based) ---")

    # Define Bayesian strategies (Non-CAPM)
    # These posterior functions are from bayesian_estimators
    # Constraint_fn (e.g., unconstrained) is from portfolio_optimizers

    bayesian_strategies_to_run = {
        "Bayesian Diffuse Prior": {
            "posterior_fn": bayesian_estimators.bayesian_diffuse_prior_moments,
            "constraint_fn": portfolio_optimizers.unconstrained, # Example constraint
            "ridge": False,
            "posterior_kwargs": {}
        },
        "Bayesian Non-info Conjugate": { # Shortened name
            "posterior_fn": bayesian_estimators.informative_bayes_predictive_niw,
            "constraint_fn": portfolio_optimizers.unconstrained,
            "ridge": False,
            "posterior_kwargs": { # For non-informative NIW
                "use_flat_mean_prior": True,
                "tau_prior_confidence": 1e-6, # Minimal confidence in flat prior
                "use_flat_cov_prior": True
            }
        },
        # Example of a history-informed prior strategy, if historical data is good
        # "Bayesian Hist-info Conjugate": {
        #     "posterior_fn": bayesian_estimators.informative_bayes_predictive_niw,
        #     "constraint_fn": portfolio_optimizers.unconstrained,
        #     "ridge": False,
        #     "posterior_kwargs": {
        #         "eta_prior_mean": historical_excess_returns_prior_df.mean().values if not historical_excess_returns_prior_df.empty else None,
        #         "tau_prior_confidence": 15, # As per notebook example
        #         "nu_prior_dof": float(historical_excess_returns_prior_df.shape[1] + 2) if not historical_excess_returns_prior_df.empty else None,
        #         "omega_prior_scale_matrix": sample_historical_covariance_prior * (float(historical_excess_returns_prior_df.shape[1] + 2) - historical_excess_returns_prior_df.shape[1] - 1) if not sample_historical_covariance_prior.empty else None,
        #         "use_flat_mean_prior": False, # Indicate that eta is provided
        #         "use_flat_cov_prior": False  # Indicate that omega is provided
        #     }
        # },
        # Ridge versions from original notebook (cell In[78])
        "Bayesian Diffuse Prior Ridge": {
            "posterior_fn": bayesian_estimators.bayesian_diffuse_prior_moments,
            "constraint_fn": portfolio_optimizers.unconstrained_ridge, # Use ridge constraint
            "ridge": True, # Explicitly pass ridge flag to solver
            "lambda_ridge": 0.1, # Default, make configurable if needed
            "posterior_kwargs": {}
        },
        "Bayesian Non-info Conjugate Ridge": {
            "posterior_fn": bayesian_estimators.informative_bayes_predictive_niw,
            "constraint_fn": portfolio_optimizers.unconstrained_ridge,
            "ridge": True,
            "lambda_ridge": 0.1,
            "posterior_kwargs": {
                "use_flat_mean_prior": True,
                "tau_prior_confidence": 1e-6,
                "use_flat_cov_prior": True
            }
        }
    }

    bayesian_results_collection = {}
    bayesian_cumulative_returns_collection = {}
    bayesian_weights_collection = {}

    for strat_name, config in bayesian_strategies_to_run.items():
        print(f"\nRunning Bayesian strategy: {strat_name}...")

        # Skip history-informed if data is missing
        if "Hist-info" in strat_name and (historical_excess_returns_prior_df.empty or sample_historical_covariance_prior.empty):
            print(f"Skipping {strat_name} due to missing historical prior data.")
            continue

        # Clean posterior_kwargs for the 'Hist-info' case if any prior components are None
        current_posterior_kwargs = config["posterior_kwargs"].copy()
        if "Hist-info" in strat_name:
            if current_posterior_kwargs.get("eta_prior_mean") is None or \
               current_posterior_kwargs.get("nu_prior_dof") is None or \
               current_posterior_kwargs.get("omega_prior_scale_matrix") is None:
                print(f"Skipping {strat_name} due to incomplete prior components from historical data.")
                continue
            # Ensure omega is correctly dimensioned if assets changed
            if not excess_returns_df.columns.equals(historical_excess_returns_prior_df.columns):
                 # This case needs careful handling of prior alignment if assets differ significantly.
                 # For now, the NIW function itself might need to handle sub-selection based on current est_data assets.
                 # The current setup for NIW prior assumes priors match the assets in excess_returns_df.
                 # A more robust way would be to align priors inside the posterior_fn or here.
                 pass


        weights_df, strategy_excess_returns_series = backtesting_engine.run_bayesian_strategy(
            excess_returns_df=excess_returns_df,
            estimation_window=ESTIMATION_WINDOW,
            holding_period=HOLDING_PERIOD,
            risk_limit_target_vol=RISK_LIMIT_TARGET_VOL,
            constraint_fn=config["constraint_fn"],
            posterior_fn=config["posterior_fn"],
            portfolio_solver_fn=portfolio_optimizers.solve_portfolio,
            ridge=config.get("ridge", False),
            lambda_ridge=config.get("lambda_ridge", 0.1),
            **current_posterior_kwargs
        )

        summary_df, allocation_diag_df, cum_total_returns_rebased100, _ = \
            backtesting_engine.summarize_performance(
                weights_df,
                strategy_excess_returns_series,
                risk_free_aligned_for_total_ret=risk_free_series
            )

        bayesian_results_collection[strat_name] = summary_df
        bayesian_cumulative_returns_collection[strat_name] = cum_total_returns_rebased100
        bayesian_weights_collection[strat_name] = weights_df

        print(f"Summary for {strat_name}:")
        print(summary_df)
        summary_df.to_csv(os.path.join(RESULTS_DIR, f"summary_bayesian_{strat_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"))
        weights_df.to_csv(os.path.join(RESULTS_DIR, f"weights_bayesian_{strat_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"))

        avg_weights_this_strategy = allocation_diag_df["Average Weight"]
        plotting_utils.plot_average_weights_bar(
            avg_weights_this_strategy,
            strategy_name=strat_name,
            output_path=os.path.join(PLOTS_DIR, f"avg_weights_bayesian_{strat_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
        )

    # Plot comparison of cumulative returns for Bayesian strategies
    if bayesian_cumulative_returns_collection:
        # Use the same aligned benchmark as for classical strategies for consistency
        plotting_utils.plot_strategy_cumulative_returns_comparison(
            bayesian_cumulative_returns_collection,
            benchmark_series=benchmark_cum_total_for_plot, # from classical section
            benchmark_label="S&P 500",
            title="Cumulative Value: Bayesian (Non-CAPM) Strategies vs Benchmark",
            output_path=os.path.join(PLOTS_DIR, "cumulative_returns_bayesian_strategies.png")
        )
        print("Bayesian (Non-CAPM) strategies cumulative returns comparison plot generated.")

        # Consolidate Bayesian summaries for metrics plot
        all_bayesian_summaries_df = pd.concat(
            [df.assign(Strategy=name) for name, df in bayesian_results_collection.items()]
        ).set_index("Strategy")

        # Combine with S&P 500 benchmark data if available
        if 'benchmark_summary_for_plot' in locals() and benchmark_summary_for_plot is not None:
             all_bayesian_summaries_df = pd.concat([all_bayesian_summaries_df, benchmark_summary_for_plot])

        # Use the same metrics_to_plot and maximize_metrics as classical for consistency
        plotting_utils.plot_strategy_metrics_comparison_bar(
            all_bayesian_summaries_df,
            metrics_to_plot=metrics_to_plot_classical, # from classical section
            benchmark_name="S&P 500",
            maximize_metrics=maximize_metrics_classical, # from classical section
            output_path=os.path.join(PLOTS_DIR, "metrics_comparison_bayesian_strategies.png")
        )
        print("Bayesian (Non-CAPM) strategies metrics comparison plot generated.")

    # Combined plot of Classical Unconstrained vs Bayesian strategies (Original Cell In[79])
    # We need: classical_cumulative_returns_collection["Unconstrained"]
    # and all from bayesian_cumulative_returns_collection

    plot_classical_vs_bayes_cum_returns = {}
    if "Unconstrained" in classical_cumulative_returns_collection and not classical_cumulative_returns_collection["Unconstrained"].empty:
        plot_classical_vs_bayes_cum_returns["Classical Unconstrained"] = classical_cumulative_returns_collection["Unconstrained"]
    if "Unconstrained Ridge" in classical_cumulative_returns_collection and not classical_cumulative_returns_collection["Unconstrained Ridge"].empty:
        plot_classical_vs_bayes_cum_returns["Classical Unconstrained Ridge"] = classical_cumulative_returns_collection["Unconstrained Ridge"]

    for name, series in bayesian_cumulative_returns_collection.items():
        if not series.empty:
            plot_classical_vs_bayes_cum_returns[name] = series

    if plot_classical_vs_bayes_cum_returns:
        plotting_utils.plot_strategy_cumulative_returns_comparison(
            plot_classical_vs_bayes_cum_returns,
            benchmark_series=benchmark_cum_total_for_plot,
            benchmark_label="S&P 500",
            title="Cumulative Value: Classical Unconstrained vs. Bayesian (Non-CAPM) Strategies",
            output_path=os.path.join(PLOTS_DIR, "cumulative_returns_classical_vs_bayesian.png")
        )
        print("Classical vs Bayesian (Non-CAPM) cumulative returns comparison plot generated.")

    # --- 10. Bayesian CAPM Framework & Efficient Frontier ---
    print("\n--- Section 10: Bayesian CAPM Framework & Efficient Frontier ---")

    # Ensure benchmark_excess_returns_series is available (loaded in Section 1)
    if 'benchmark_excess_returns_series' not in locals() or benchmark_excess_returns_series.empty:
        print("Benchmark excess returns not available, skipping Bayesian CAPM section.")
    else:
        # Calculate Bayesian CAPM predictive moments (Original cell In[88])
        mu_bcapm, sigma_bcapm = bayesian_estimators.bayesian_capm_estimation(
            excess_returns_df, benchmark_excess_returns_series
            # Use default B-CAPM priors
            # These can be exposed as config variables if tuning is needed.
        )

        # Compute efficient frontier using these moments
        tr_bcapm, pv_bcapm, _ = portfolio_optimizers.compute_efficient_frontier_from_moments(
            mu_bcapm, sigma_bcapm
        )

        bayesian_capm_frontier_plot_data = []
        valid_bcapm_frontier = ~np.isnan(pv_bcapm) & ~np.isnan(tr_bcapm)

        if np.sum(valid_bcapm_frontier) > 1:
            # Calculate GMV/Tangency for this frontier
            gmv_idx_bcapm = np.nanargmin(pv_bcapm[valid_bcapm_frontier])
            gmv_pt_bcapm = (pv_bcapm[valid_bcapm_frontier][gmv_idx_bcapm], tr_bcapm[valid_bcapm_frontier][gmv_idx_bcapm])

            sharpe_bcapm = (tr_bcapm[valid_bcapm_frontier] - rf_for_tangency) / pv_bcapm[valid_bcapm_frontier] # rf_for_tangency from Sec 6
            tan_idx_bcapm = np.nanargmax(sharpe_bcapm)
            tan_pt_bcapm = (pv_bcapm[valid_bcapm_frontier][tan_idx_bcapm], tr_bcapm[valid_bcapm_frontier][tan_idx_bcapm])

            bayesian_capm_frontier_plot_data.append(
                (pv_bcapm[valid_bcapm_frontier],
                 tr_bcapm[valid_bcapm_frontier],
                 "Bayesian CAPM",
                 gmv_pt_bcapm, tan_pt_bcapm,
                 {'line_color': 'navy', 'dot_color': 'blue', 'line_style': '-'})
            )

            plotting_utils.plot_efficient_frontier(
                bayesian_capm_frontier_plot_data,
                title="Efficient Frontier (Bayesian CAPM Predictive Inputs)",
                # xlim, ylim can be set if desired, or let matplotlib auto-scale
                output_path=os.path.join(PLOTS_DIR, "efficient_frontier_bayesian_capm.png")
            )
            print("Bayesian CAPM efficient frontier plot generated.")
        else:
            print("Not enough valid points to plot Bayesian CAPM efficient frontier.")

    # --- 11. Bayesian CAPM-based Rolling Window Strategies ---
    print("\n--- Section 11: Bayesian CAPM-based Rolling Window Strategies ---")

    if 'benchmark_excess_returns_series' not in locals() or benchmark_excess_returns_series.empty:
        print("Benchmark excess returns not available, skipping Bayesian CAPM-based strategies.")
    else:
        # Define Bayesian CAPM-based strategies (Original cell In[93])
        # Posterior functions are from bayesian_estimators
        # gmv_vol_calculator_fn for adaptive risk is from portfolio_optimizers
        bayesian_capm_strategies_to_run = {
            "MVO (Bayesian CAPM moments)": { # MVO (B-CAPM moments)
                "posterior_fn": bayesian_estimators.bayesian_capm_posterior_moments,
                "constraint_fn": portfolio_optimizers.unconstrained,
                "ridge": False,
                "adaptive_risk": True, # As per notebook cell In[94] adaptive_risk=True
                "posterior_kwargs": {}
            },
            "NIW (Bayesian CAPM prior)": { # NIW (B-CAPM prior)
                "posterior_fn": bayesian_estimators.bayes_conjugate_with_capm_prior_moments,
                "constraint_fn": portfolio_optimizers.unconstrained,
                "ridge": False,
                "adaptive_risk": True,
                "posterior_kwargs": { # These are passed to bayes_conjugate_with_capm_prior_moments
                    "capm_prior_confidence_tau": 10.0, # Default from function
                    "capm_prior_dof_nu_add": 10.0    # Default from function
                    # capm_kwargs for the internal bayesian_capm_estimation can also be passed here if needed
                }
            },
            "NIW (Bayesian CAPM prior) Ridge": {
                "posterior_fn": bayesian_estimators.bayes_conjugate_with_capm_prior_moments,
                "constraint_fn": portfolio_optimizers.unconstrained_ridge, # Ridge constraint
                "ridge": True, # Enable ridge in solver
                "lambda_ridge": 0.1,
                "adaptive_risk": True,
                "posterior_kwargs": {
                    "capm_prior_confidence_tau": 10.0,
                    "capm_prior_dof_nu_add": 10.0
                }
            }
        }

        bayesian_capm_results_collection = {}
        bayesian_capm_cumulative_returns_collection = {}
        bayesian_capm_weights_collection = {}

        for strat_name, config in bayesian_capm_strategies_to_run.items():
            print(f"\nRunning Bayesian CAPM-based strategy: {strat_name}...")

            weights_df, strategy_excess_returns_series = \
                backtesting_engine.run_custom_bayesian_strategy_with_benchmark(
                    excess_returns_df=excess_returns_df,
                    benchmark_returns_series=benchmark_excess_returns_series, # Crucial for these strategies
                    estimation_window=ESTIMATION_WINDOW,
                    holding_period=HOLDING_PERIOD,
                    risk_limit_target_vol=RISK_LIMIT_TARGET_VOL, # Fixed target for adaptive risk base
                    constraint_fn=config["constraint_fn"],
                    posterior_fn=config["posterior_fn"],
                    portfolio_solver_fn=portfolio_optimizers.solve_portfolio,
                    gmv_vol_calculator_fn=portfolio_optimizers.compute_gmv_vol if config.get("adaptive_risk", False) else None,
                    ridge=config.get("ridge", False),
                    lambda_ridge=config.get("lambda_ridge", 0.1),
                    adaptive_risk_target_vol=RISK_LIMIT_TARGET_VOL if config.get("adaptive_risk", False) else None,
                    **config["posterior_kwargs"]
                )

            summary_df, allocation_diag_df, cum_total_returns_rebased100, _ = \
                backtesting_engine.summarize_performance(
                    weights_df,
                    strategy_excess_returns_series,
                    risk_free_aligned_for_total_ret=risk_free_series
                )

            bayesian_capm_results_collection[strat_name] = summary_df
            bayesian_capm_cumulative_returns_collection[strat_name] = cum_total_returns_rebased100
            bayesian_capm_weights_collection[strat_name] = weights_df

            print(f"Summary for {strat_name}:")
            print(summary_df)
            summary_df.to_csv(os.path.join(RESULTS_DIR, f"summary_bcapm_{strat_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"))
            weights_df.to_csv(os.path.join(RESULTS_DIR, f"weights_bcapm_{strat_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"))

            avg_weights_this_strategy = allocation_diag_df["Average Weight"]
            plotting_utils.plot_average_weights_bar(
                avg_weights_this_strategy,
                strategy_name=strat_name,
                output_path=os.path.join(PLOTS_DIR, f"avg_weights_bcapm_{strat_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
            )

        # Plot comparison of cumulative returns for Bayesian CAPM strategies
        if bayesian_capm_cumulative_returns_collection:
            plotting_utils.plot_strategy_cumulative_returns_comparison(
                bayesian_capm_cumulative_returns_collection,
                benchmark_series=benchmark_cum_total_for_plot, # from classical section
                benchmark_label="S&P 500",
                title="Cumulative Value: Bayesian CAPM-based Strategies vs Benchmark",
                output_path=os.path.join(PLOTS_DIR, "cumulative_returns_bayesian_capm_strategies.png")
            )
            print("Bayesian CAPM-based strategies cumulative returns comparison plot generated.")

            # Consolidate Bayesian CAPM summaries for metrics plot
            all_bcapm_summaries_df = pd.concat(
                [df.assign(Strategy=name) for name, df in bayesian_capm_results_collection.items()]
            ).set_index("Strategy")

            if 'benchmark_summary_for_plot' in locals() and benchmark_summary_for_plot is not None:
                 all_bcapm_summaries_df = pd.concat([all_bcapm_summaries_df, benchmark_summary_for_plot])

            plotting_utils.plot_strategy_metrics_comparison_bar(
                all_bcapm_summaries_df,
                metrics_to_plot=metrics_to_plot_classical, # from classical section
                benchmark_name="S&P 500",
                maximize_metrics=maximize_metrics_classical, # from classical section
                output_path=os.path.join(PLOTS_DIR, "metrics_comparison_bayesian_capm_strategies.png")
            )
            print("Bayesian CAPM-based strategies metrics comparison plot generated.")

    # --- 12. Final Summary Plots & Data Export ---
    print("\n--- Section 12: Final Summary Plots & Data Export ---")

    # Grand comparison of cumulative returns (Original cell In[98])
    final_cumulative_returns_summary_plot = {}
    # Add selected classical strategies
    if "Unconstrained" in classical_cumulative_returns_collection:
        final_cumulative_returns_summary_plot["Classical Unconstrained"] = classical_cumulative_returns_collection["Unconstrained"]
    if "Unconstrained Ridge" in classical_cumulative_returns_collection:
        final_cumulative_returns_summary_plot["Classical Unconstrained Ridge"] = classical_cumulative_returns_collection["Unconstrained Ridge"]
    if "Long Only" in classical_cumulative_returns_collection: # As per original plot
        final_cumulative_returns_summary_plot["Classical Long Only"] = classical_cumulative_returns_collection["Long Only"]

    # Add selected Bayesian (non-CAPM) strategies
    for name in ["Bayesian Diffuse Prior", "Bayesian Non-info Conjugate",
                 "Bayesian Diffuse Prior Ridge", "Bayesian Non-info Conjugate Ridge"]:
        if name in bayesian_cumulative_returns_collection:
            final_cumulative_returns_summary_plot[name] = bayesian_cumulative_returns_collection[name]

    # Add selected Bayesian CAPM-based strategies
    for name in bayesian_capm_cumulative_returns_collection: # Add all from this category
        final_cumulative_returns_summary_plot[name] = bayesian_capm_cumulative_returns_collection[name]

    if final_cumulative_returns_summary_plot:
        plotting_utils.plot_strategy_cumulative_returns_comparison(
            final_cumulative_returns_summary_plot,
            benchmark_series=benchmark_cum_total_for_plot,
            benchmark_label="S&P 500",
            title="Final Cumulative Portfolio Value of All Key Strategies vs Benchmark",
            output_path=os.path.join(PLOTS_DIR, "cumulative_returns_all_strategies_summary.png")
        )
        print("Final all-strategies cumulative returns summary plot generated.")

    # Consolidate ALL strategy summaries for a final table and potential detailed metrics plot
    all_summaries_list = []
    for name, df in classical_results_collection.items(): all_summaries_list.append(df.assign(Strategy=name))
    for name, df in bayesian_results_collection.items(): all_summaries_list.append(df.assign(Strategy=name))
    for name, df in bayesian_capm_results_collection.items(): all_summaries_list.append(df.assign(Strategy=name))

    if all_summaries_list:
        grand_summary_table = pd.concat(all_summaries_list).set_index("Strategy")
        if 'benchmark_summary_for_plot' in locals() and benchmark_summary_for_plot is not None:
            grand_summary_table = pd.concat([grand_summary_table, benchmark_summary_for_plot])

        print("\nFinal Grand Summary Table of All Strategies:")
        print(grand_summary_table.round(4))
        grand_summary_table.to_csv(os.path.join(RESULTS_DIR, "grand_summary_all_strategies.csv"))
        print("Grand summary table saved to CSV.")

        # Final metrics plot (using the same metrics as before)
        plotting_utils.plot_strategy_metrics_comparison_bar(
            grand_summary_table,
            metrics_to_plot=metrics_to_plot_classical,
            benchmark_name="S&P 500",
            maximize_metrics=maximize_metrics_classical,
            output_path=os.path.join(PLOTS_DIR, "metrics_comparison_all_strategies_summary.png")
        )
        print("Final all-strategies metrics comparison plot generated.")


    # Replace the placeholder print statement
    # Find: print("\n(Further sections of analysis to be implemented)")
    # Replace with: print("\nAll analytical sections complete.")
    # This will be handled by ensuring the placeholder is removed from the file if it exists,
    # and this new content is the last part of the main() function before its end.
    # The subtask will append, and then I'll run another subtask to do the find/replace if necessary,
    # or simply construct the final file content carefully.
    # For now, this block is the last analytical block.

    print("\nAll analytical sections complete.")

    print("\nMVO Analysis Workflow Complete.")

if __name__ == '__main__':
    main()

