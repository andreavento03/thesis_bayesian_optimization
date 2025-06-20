#!/usr/bin/env python
# coding: utf-8

# Classical MVO



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
import scipy.stats as stats
import seaborn as sns
import cvxpy as cp

# Dataset Loading # Dataset Loading # Dataset Loading # ## Dataset Loading and Inspection of comulative returns Inspection Inspection Inspection




stock_returns = pd.read_csv("../Data/final_dataset_stocks.csv", index_col=0, parse_dates=True)
bond_returns = pd.read_csv("../Data/final_dataset_bonds.csv", index_col=0, parse_dates=True)
sp_benchmark = pd.read_csv("../Data/SP_500_benchmark.csv", index_col=0, parse_dates=True)



sp_benchmark



bond_returns = bond_returns.drop(columns=["b1ret"])
bond_returns = bond_returns.drop(columns=["b7ret"])


bond_returns.columns.name = None
bond_returns.index.name = None


bond_returns.index = pd.to_datetime(bond_returns.index)

bond_returns = bond_returns.apply(pd.to_numeric)
bond_returns



stock_returns = stock_returns.copy()
bond_returns = bond_returns.copy()

stock_returns = stock_returns.reset_index().rename(columns={'index': 'Date'})
bond_returns = bond_returns.reset_index().rename(columns={'index': 'Date'})


returns = pd.merge(stock_returns, bond_returns, on='Date', how='inner')

returns.set_index('Date', inplace=True)
returns.index.name = None

returns[0:15]



# ticker map

column_names = list(returns.columns)
column_names

name_to_ticker = {
    'A T & T INC': 'T',
    'ALPHABET INC': 'GOOGL',
    'ALTRIA GROUP INC': 'MO',
    'AMERICAN INTERNATIONAL GROUP INC': 'AIG',
    'APPLE INC': 'AAPL',
    'BANK OF AMERICA CORP': 'BAC',
    'BERKSHIRE HATHAWAY INC DEL': 'BRK.B',
    'CHEVRON CORP NEW': 'CVX',
    'CISCO SYSTEMS INC': 'CSCO',
    'CITIGROUP INC': 'C',
    'COCA COLA CO': 'KO',
    'EXXON MOBIL CORP': 'XOM',
    'GENERAL ELECTRIC CO': 'GE',
    'INTEL CORP': 'INTC',
    'INTERNATIONAL BUSINESS MACHS COR': 'IBM',
    'JOHNSON & JOHNSON': 'JNJ',
    'MICROSOFT CORP': 'MSFT',
    'PFIZER INC': 'PFE',
    'PROCTER & GAMBLE CO': 'PG',
    'WALMART INC': 'WMT',
    'b20ret': 'BOND20'
}

# rename columns
returns.rename(columns=name_to_ticker, inplace=True)



returns

# S# S# S# S# Sp 500 benchmark metricsP 500 metricsP 500 metricsP 500 metricsP 500 metrics



# align benchmark
sp_benchmark = sp_benchmark.loc[returns.index]

sp500_mean = sp_benchmark["sprtrn"].mean()
sp500_std = sp_benchmark["sprtrn"].std()
sp500_var = sp_benchmark["sprtrn"].var()

benchmark_returns = sp_benchmark["sprtrn"].loc[returns.index]

cumulative_return_sp500_total = (1 + benchmark_returns).prod() - 1
mean_monthly_sp500_total = benchmark_returns.mean()
volatility_sp500 = benchmark_returns.std()
sharpe_sp500 = mean_monthly_sp500_total / volatility_sp500 * np.sqrt(12)
annualized_return_sp500_total = ((1 + mean_monthly_sp500_total) ** 12 - 1) * 100



mean_monthly_sp500_total

# Cumulative Return: Portfolio vs Benchmark



# cumulative wealth ($100)
wealth = (1 + returns).cumprod() * 100
benchmark_wealth = (1 + sp_benchmark["sprtrn"]).cumprod() * 100

plt.figure(figsize=(14, 6))

# plot asset inception
for column in wealth.columns:
    series = wealth[column].dropna()
    plt.plot(series.index, series.values, label=column, linewidth=1)

# plot benchmark (thick black)
plt.plot(benchmark_wealth.index, benchmark_wealth,
         label="S&P 500 Benchmark", linewidth=3, color='black')

plt.title("Cumulative Wealth per Company vs. S&P 500 Benchmark (Initial $100 Investment)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# cumulative wealth ($100)
wealth = (1 + returns).cumprod() * 100
benchmark_wealth = (1 + sp_benchmark["sprtrn"]).cumprod() * 100

plt.figure(figsize=(16, 7))
ax = plt.gca()

# top/bottom performers
final_wealth = wealth.iloc[-1].sort_values(ascending=False)
best_label = final_wealth.index[0]
second_label = final_wealth.index[1]
third_label = final_wealth.index[2]
second_worst_label = final_wealth.index[-2]
worst_label = final_wealth.index[-1]

major_labels = [best_label, second_label, third_label, second_worst_label, worst_label]

# colors for plot
label_color_map = {
    best_label: 'green',           # best
    second_label: 'blue',          # second
    third_label: 'yellow',         # third
    second_worst_label: 'orange',  # second worst
    worst_label: 'black'           # worst
}
sp_label = "S&P 500 Benchmark"

# legend lines
line_handles = {}

# plot assets (highlight key ones)
for column in wealth.columns:
    series = wealth[column].dropna()
    if column in major_labels:
        color = label_color_map[column]
        line, = ax.plot(series.index, series.values,
                        linewidth=2.2, alpha=0.96, label=column, zorder=2, color=color)
        line_handles[column] = line
    else:
        ax.plot(series.index, series.values,
                linewidth=1, alpha=0.55, color='grey', zorder=1)

# plot S# plot S# Plot S&P 500 benchmark with thickest line and red colorP500 (thick red)P500 (thick red)
sp500_line, = ax.plot(
    benchmark_wealth.index, benchmark_wealth,
    linewidth=3.5, color='red', alpha=0.98,
    label=sp_label, zorder=3
)

ax.set_title("Cumulative Wealth per Company vs. S&P 500 Benchmark (Initial $100 Investment)", fontsize=15, pad=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Portfolio Value ($)", fontsize=12)
ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

# custom legend
legend_handles = [sp500_line] + [line_handles[label] for label in major_labels]
legend_labels = [sp_label] + major_labels
ax.legend(legend_handles, legend_labels, loc='upper left', fontsize='medium', frameon=True)

plt.tight_layout()
plt.show()




# cumulative return table
cumulative_returns_assets = (1 + returns).prod() - 1
cumulative_returns_assets_percent = cumulative_returns_assets * 100
cumulative_returns_assets_table = cumulative_returns_assets_percent.round(2).to_frame(name="Cumulative Return (%)")

# benchmark cumulative return
benchmark_return = (1 + sp_benchmark["sprtrn"]).prod() - 1
benchmark_return_percent = benchmark_return * 100
benchmark_return_table = pd.DataFrame({"Cumulative Return (%)": [round(benchmark_return_percent, 2)]},
                                      index=["S&P 500 Benchmark"])

# combined summary table
combined_cumulative_returns = pd.concat([cumulative_returns_assets_table, benchmark_return_table])
combined_cumulative_returns = combined_cumulative_returns.sort_values(by="Cumulative Return (%)", ascending=False)

combined_cumulative_returns



# mean monthly returns
mean_monthly_returns = returns.mean()

# benchmark mean monthly return
benchmark_mean_monthly_return = sp_benchmark["sprtrn"].loc[returns.index].mean()

# annualized returns
annualized_returns_assets = ((1 + mean_monthly_returns) ** 12 - 1) * 100
annualized_return_benchmark = ((1 + benchmark_mean_monthly_return) ** 12 - 1) * 100

# combine returns (sp500 label)
annualized_returns_combined = annualized_returns_assets.copy()
annualized_returns_combined["sp500"] = annualized_return_benchmark

# sort desc
annualized_returns_combined = annualized_returns_combined.sort_values(ascending=False)

# colors: sp500 red, others blue
colors = ['red' if stock == 'sp500' else 'skyblue' for stock in annualized_returns_combined.index]

plt.figure(figsize=(14, 6))
annualized_returns_combined.plot(
    kind='bar',
    color=colors,
    edgecolor='black'
)
plt.title("Annualized Return per Stock and S&P 500 Benchmark", fontsize=14, weight='bold')
plt.xlabel("Asset", fontsize=12)
plt.ylabel("Annualized Return (%)", fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


# Excess Returns



rf = pd.read_csv("/Users/andreavento/Documents/Tesi/Dataset/3-month_T_bill.csv")
rf



rf.columns = ['Date', 'Yield']
rf['Date'] = pd.to_datetime(rf['Date'])
rf.set_index('Date', inplace=True)
rf = rf.dropna()
rf




# monthly yields (end of month)
rf_monthly = rf.resample('M').last()

# shift for fwd-looking window
rf_monthly_shifted = rf_monthly.shift(1)

# ann. yield to monthly return (cont.)
rf_monthly_shifted['rf_return'] = np.exp(rf_monthly_shifted['Yield'] / 100 / 12) - 1

# date range
start_date = "2008-01-31"
end_date = "2022-12-31"

# filter rf series by date
rf_final = rf_monthly_shifted.loc[start_date:end_date]

rf_final



rf_final_aligned = rf_final.reindex(returns.index, method='ffill')
rf_final_aligned



# compute excess returns
excess_returns = returns.sub(rf_final_aligned['rf_return'], axis=0)

excess_returns

# Benchmark metrics (excess return adj.)



benchmark_rf = rf_final_aligned['rf_return']
benchmark_excess_returns = benchmark_returns - benchmark_rf
cumulative_return_sp500 = (1 + benchmark_returns).prod() - 1

# excess return metrics
mean_monthly_sp500 = benchmark_excess_returns.mean()
volatility_sp500 = benchmark_excess_returns.std()
sharpe_sp500 = mean_monthly_sp500 / volatility_sp500 * np.sqrt(12)
annualized_return_sp500 = ((1 + mean_monthly_sp500) ** 12 - 1) * 100

benchmark_metrics = pd.DataFrame({
    "Total Cumulative Return": [cumulative_return_sp500],
    "Mean Monthly Return": [mean_monthly_sp500],
    "Volatility (Monthly Std)": [volatility_sp500],
    "Annualized Sharpe Ratio": [sharpe_sp500],
    "Annualized Return %": [annualized_return_sp500],
    "Average HHI": [np.nan],
    "HHI Std Dev": [np.nan],
    "Avg Rolling Weight Std Dev": [np.nan],
    "Average Turnover": [0],
    "Turnover Std Dev": [0]
}, index=["S&P 500"])

# Historical Excess Returns (2000-2007)



historical_df = pd.read_csv('../Data/historical_before2008_returns.csv', index_col=0, parse_dates=True)
historical_df = historical_df[24::].copy()
historical_df = historical_df[returns.columns]
historical_df

# Google NAs (pre-IPO)

# RF for period



rf_1 = pd.read_csv('../Data/DGS1MO-2.csv')
rf_1



rf_1.columns = ['Date', 'Yield']
rf_1['Date'] = pd.to_datetime(rf_1['Date'])
rf_1.set_index('Date', inplace=True)
rf_1 = rf_1.dropna()
rf_1




# monthly yields (end of month)
rf_1_monthly = rf_1.resample('M').last()

# shift for fwd-looking window
rf_1_monthly_shifted = rf_1_monthly.shift(1)

# ann. yield to monthly return (cont.)
rf_1_monthly_shifted['rf_return'] = np.exp(rf_1_monthly_shifted['Yield'] / 100 / 12) - 1

# date range
start_date_1 = "2002-01-31"
end_date_1 = "2007-12-31"

# filter rf series by date
rf_1_final = rf_1_monthly_shifted.loc[start_date_1:end_date_1]

rf_final_1_aligned = rf_1_final.reindex(historical_df.index, method='ffill')
rf_final_1_aligned


# historical excess returns



# compute excess returns
historical_excess_returns = historical_df.sub(rf_final_1_aligned['rf_return'], axis=0)

historical_excess_returns




# avg mean/var of avg asset (hist)
average_mean_historical_pre2008 = historical_excess_returns.mean(axis=1).mean()
average_variance_historical_pre2008 = historical_excess_returns.var(axis=1).mean()
average_mean_historical_pre2008, average_variance_historical_pre2008

# sample hist cov matrix
sample_historical_covariance = historical_excess_returns.cov()
sample_historical_covariance

# Hist. S# Hist. S# Hist. S# Hist. S# Historical SP500 returns (2002-2007)P500 (2002-2007)P500 (2002-2007)P500 (2002-2007)P500 (2002-2007)




sp500_pre = pd.read_csv('../Data/SP_500_benchmark_presample.csv', index_col=0, parse_dates=True)

# filter 2002-2007
sp500_pre_filtered = sp500_pre.loc['2002-01-31':'2007-12-31']
sp500_pre_filtered = sp500_pre_filtered.rename(columns={'sprtrn': 'sp500_return'})
sp500_pre_filtered




excess_sp500_pre = sp500_pre_filtered.sub(rf_final_1_aligned['rf_return'], axis=0)


excess_sp500_pre.columns.name = None
excess_sp500_pre.index.name = None

# ensure datetime index
excess_sp500_pre.index = pd.to_datetime(excess_sp500_pre.index)
excess_sp500_pre

# EDA



# 1. compute metrics
mean_excess = excess_returns.mean()
volatility_excess = excess_returns.std()

# add benchmark
mean_excess["S&P 500"] = benchmark_excess_returns.mean()
volatility_excess["S&P 500"] = benchmark_excess_returns.std()

# 2. Sort by mean excess return
sorted_mean = mean_excess.sort_values(ascending=False)
sorted_volatility = volatility_excess[sorted_mean.index]

# 3. Define color scheme
colors = ['red' if asset == 'S&P 500' else 'skyblue' for asset in sorted_mean.index]

# 4. Plot mean excess returns
plt.figure(figsize=(14, 6))
sorted_mean.plot(kind='bar', color=colors, edgecolor='black')
plt.title("Mean Monthly Excess Return: Assets vs. S&P 500")
plt.ylabel("Mean Excess Return")
plt.xlabel("Asset")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 5. Plot volatility of excess returns
plt.figure(figsize=(14, 6))
sorted_volatility.plot(kind='bar', color=colors, edgecolor='black')
plt.title("Volatility of Excess Returns: Assets vs. S&P 500")
plt.ylabel("Standard Deviation")
plt.xlabel("Asset")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()




# 1. Create equal-weighted portfolio from excess returns
equal_weights = np.repeat(1 / excess_returns.shape[1], excess_returns.shape[1])
portfolio_excess_returns = excess_returns.dot(equal_weights)

# 2. Compute mean and std deviation (monthly)
portfolio_mean = portfolio_excess_returns.mean()
portfolio_std = portfolio_excess_returns.std()

benchmark_mean = benchmark_excess_returns.mean()
benchmark_std = benchmark_excess_returns.std()

# 3. Create summary DataFrame
comparison_df = pd.DataFrame({
    'Mean Monthly Excess Return': [portfolio_mean, benchmark_mean],
    'Standard Deviation': [portfolio_std, benchmark_std]
}, index=['Average asset', 'S&P 500'])

# 4. Plot - Mean Monthly Excess Return
plt.figure(figsize=(8, 5))
comparison_df['Mean Monthly Excess Return'].plot(kind='bar', color=['steelblue', 'red'], edgecolor='black')
plt.title("Mean Monthly Excess Return: Average asset vs. S&P 500")
plt.ylabel("Mean Excess Return")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 5. Plot - Standard Deviation
plt.figure(figsize=(8, 5))
comparison_df['Standard Deviation'].plot(kind='bar', color=['steelblue', 'red'], edgecolor='black')
plt.title("Volatility of Excess Returns: Average asset vs. S&P 500")
plt.ylabel("Standard Deviation")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Markowitz issues
# - Treats estimated inputs as true parameters (estimation risk).
# - Sensitive to input noise, leading to unstable portfolios.


# Normality
# - MVO assumes normal returns, ignoring fat tails/skewness.
# - Underestimates tail risk.




# log returns
returns_1 = returns.copy()

# Plot histograms
# plot grid setup
num_assets = returns_1.shape[1]
ncols = 10
nrows = int(np.ceil(num_assets / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 2.5))
axes = axes.flatten()

for i, col in enumerate(returns_1.columns):
    data = returns_1[col].dropna()
    ax = axes[i]

    # Histogram
    ax.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')

    # Normal distribution overlay
    mu, std = data.mean(), data.std()
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=1.5, label='Normal')

    # KDE overlay (best empirical fit in red)
    kde = stats.gaussian_kde(data)
    ax.plot(x, kde(x), 'r-', linewidth=1.5, label='KDE')

    ax.set_title(col, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # Optional: show legend on first few plots
    if i < 3:
        ax.legend(fontsize=6)

# hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Histograms of Log Returns with Normal and KDE (Red) Overlays", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




# log returns
log_returns = np.log(1 + returns)

# Plot histograms
# plot grid setup
num_assets = log_returns.shape[1]
ncols = 10
nrows = int(np.ceil(num_assets / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 2.5))
axes = axes.flatten()

for i, col in enumerate(log_returns.columns):
    data = log_returns[col].dropna()
    ax = axes[i]

    # Histogram
    ax.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')

    # Normal distribution overlay
    mu, std = data.mean(), data.std()
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=1.5, label='Normal')

    # KDE overlay (best empirical fit in red)
    kde = stats.gaussian_kde(data)
    ax.plot(x, kde(x), 'r-', linewidth=1.5, label='KDE')

    ax.set_title(col, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # Optional: show legend on first few plots
    if i < 3:
        ax.legend(fontsize=6)

# hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Histograms of Log Returns with Normal and KDE (Red) Overlays", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Thesis Image: Raw/Log Returns



import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# seed for reproducibility
np.random.seed(42)
sampled_cols = np.random.choice(returns.columns, size=5, replace=False)

# setup figure
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 5))
axes = axes.flatten()

for i, col in enumerate(sampled_cols):
    # Raw returns
    data_raw = returns[col].dropna()
    ax_raw = axes[i]
    ax_raw.hist(data_raw, bins=30, density=True, alpha=0.5, color='steelblue', edgecolor='black')

    mu, std = data_raw.mean(), data_raw.std()
    x = np.linspace(*ax_raw.get_xlim(), 100)
    ax_raw.plot(x, stats.norm.pdf(x, mu, std), 'k-', linewidth=1.3, label='Normal')
    ax_raw.plot(x, stats.gaussian_kde(data_raw)(x), 'r--', linewidth=1.3, label='KDE')

    ax_raw.set_title(f"Raw: {col}", fontsize=10)
    ax_raw.set_xticks([])
    ax_raw.set_yticks([])
    ax_raw.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)

    if i == 0:
        ax_raw.legend(fontsize=7, loc='upper right')

    # Log returns
    data_log = log_returns[col].dropna()
    ax_log = axes[i + 5]
    ax_log.hist(data_log, bins=30, density=True, alpha=0.5, color='steelblue', edgecolor='black')

    mu, std = data_log.mean(), data_log.std()
    x = np.linspace(*ax_log.get_xlim(), 100)
    ax_log.plot(x, stats.norm.pdf(x, mu, std), 'k-', linewidth=1.3)
    ax_log.plot(x, stats.gaussian_kde(data_log)(x), 'r--', linewidth=1.3)

    ax_log.set_title(f"Log: {col}", fontsize=10)
    ax_log.set_xticks([])
    ax_log.set_yticks([])
    ax_log.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)

fig.suptitle("Sample of Raw and Log Return Distributions for 5 assets", fontsize=14, y=0.86)
plt.tight_layout(rect=[0, 0.02, 1, 0.88])
plt.show()

# investigate normality (raw vs log)



# ticker map

column_names = list(returns.columns)
column_names

name_to_ticker = {
    'A T & T INC': 'T',
    'ALPHABET INC': 'GOOGL',
    'ALTRIA GROUP INC': 'MO',
    'AMERICAN INTERNATIONAL GROUP INC': 'AIG',
    'APPLE INC': 'AAPL',
    'BANK OF AMERICA CORP': 'BAC',
    'BERKSHIRE HATHAWAY INC DEL': 'BRK.B',
    'CHEVRON CORP NEW': 'CVX',
    'CISCO SYSTEMS INC': 'CSCO',
    'CITIGROUP INC': 'C',
    'COCA COLA CO': 'KO',
    'EXXON MOBIL CORP': 'XOM',
    'GENERAL ELECTRIC CO': 'GE',
    'INTEL CORP': 'INTC',
    'INTERNATIONAL BUSINESS MACHS COR': 'IBM',
    'JOHNSON & JOHNSON': 'JNJ',
    'MICROSOFT CORP': 'MSFT',
    'PFIZER INC': 'PFE',
    'PROCTER & GAMBLE CO': 'PG',
    'WALMART INC': 'WMT',
    'b20ret': 'BOND20'
}

# rename columns
returns.rename(columns=name_to_ticker, inplace=True)



from scipy.stats import shapiro, jarque_bera, anderson

def run_normality_tests(df, label=''):
    results = []

    for col in df.columns:
        data = df[col].dropna()

        # Shapiro-Wilk Test
        shapiro_stat, shapiro_p = shapiro(data)

        # Jarque-Bera Test
        jb_stat, jb_p = jarque_bera(data)

        # Anderson-Darling Test
        ad_result = anderson(data, dist='norm')
        ad_stat = ad_result.statistic
        ad_crit_val = ad_result.critical_values[2]  # 5% level

        results.append({
            'Asset': col,
            'Shapiro_p': shapiro_p,
            'JarqueBera_p': jb_p,
            'AD_stat': ad_stat,
            'AD_5%_crit': ad_crit_val,
            'AD_pass': ad_stat < ad_crit_val
        })

    # Compile into DataFrame
    test_results = pd.DataFrame(results)
    test_results['Shapiro_pass'] = test_results['Shapiro_p'] > 0.05
    test_results['JB_pass'] = test_results['JarqueBera_p'] > 0.05
    test_results['All_Passed'] = test_results['Shapiro_pass'] & test_results['JB_pass'] & test_results['AD_pass']
    test_results['All_Failed'] = ~test_results['Shapiro_pass'] & ~test_results['JB_pass'] & ~test_results['AD_pass']

    # Summary
    summary = {
        'Total Assets': len(test_results),
        'Shapiro Passed': test_results['Shapiro_pass'].sum(),
        'Jarque-Bera Passed': test_results['JB_pass'].sum(),
        'Anderson-Darling Passed': test_results['AD_pass'].sum(),
        'Passed All Tests': test_results['All_Passed'].sum(),
        'Failed All Tests': test_results['All_Failed'].sum(),
        'Partial Agreement': len(test_results) - test_results['All_Passed'].sum() - test_results['All_Failed'].sum()
    }

    summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=[f'{label} Count'])

    return test_results, summary_df




# Log returns
log_returns = np.log(1 + returns)

# Run for raw returns
raw_results, raw_summary = run_normality_tests(returns, label='Raw Returns')

# Run for log returns
log_results, log_summary = run_normality_tests(log_returns, label='Log Returns')

# combine summaries
combined_summary = pd.concat([raw_summary, log_summary], axis=1)

combined_summary


# Normality in Bayesian Framework:
# - Normality assumed for NIW priors (tractability, comparability).
# - Empirical returns often deviate (skewness, kurtosis).
# - Heavy-tailed priors will be explored later.

# Mean Sensitivity



mu_hat_trial = excess_returns.mean()
mu_hat_trial

# Mean Sensitivity:
# - Test MVO sensitivity to mean estimates (tangency portfolio).
# - Add Gaussian noise to simulate estimation error (Best & Grauer 1991, Chopra & Ziemba 1993).
# - Perturbations: 10-50 bps, consistent with empirical forecast errors.



# SENSITIVITY: EXPECTED RETURNS

# mean # Mean and covariance cov
mu_hat = excess_returns.mean().values
Sigma_hat = excess_returns.cov().values
assets = excess_returns.columns

def compute_mvo_weights(mu, Sigma):
    inv_Sigma = inv(Sigma)
    weights = inv_Sigma @ mu
    weights /= np.sum(weights)
    return weights


def compute_long_only_mvo_weights(mu, Sigma, risk_limit=0.03):
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    ret = mu @ w
    constraints = [cp.sum(w) == 1, w >= 0, risk <= risk_limit**2]
    problem = cp.Problem(cp.Maximize(ret), constraints)
    problem.solve()
    return w.value

# baseline weights
baseline_weights = compute_mvo_weights(mu_hat, Sigma_hat)
baseline_weights_longonly = compute_long_only_mvo_weights(mu_hat, Sigma_hat)


np.random.seed(123)
epsilon_mu = np.random.normal(loc=0, scale=0.001, size=mu_hat.shape)
mu_noisy = mu_hat + epsilon_mu


# perturbed mvo weights
weights_noisy = compute_mvo_weights(mu_noisy, Sigma_hat)
weights_noisy_longonly = compute_long_only_mvo_weights(mu_noisy, Sigma_hat)


# compare baseline # Compare baseline and noisy-mean weights noisy weights
df_compare_noise = pd.DataFrame({
    'Baseline': baseline_weights,
    'Perturbed (Gaussian noise μ+N(0, 0.001))': weights_noisy
}, index=assets)





percentual_change = np.abs((weights_noisy - baseline_weights) / np.where(baseline_weights != 0, baseline_weights, np.nan))
avg_percentual_change = np.nanmean(percentual_change) * 100

print(f"Average absolute percentual change in weights: {avg_percentual_change:.2f}%")





print(baseline_weights_longonly)



import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-muted')

# custom colors
colors = ['#4B8BBE', 'red']

ax = df_compare_noise.plot(
    kind='bar',
    figsize=(14, 6),
    width=0.8,
    alpha=0.98,
    edgecolor='k',
    color=colors
)

# smaller fonts
ax.set_title("Portfolio Weights Before and After Gaussian Noise Perturbation", fontsize=15, pad=10)
ax.set_ylabel("Portfolio Weight", fontsize=12)
ax.set_xlabel("Assets", fontsize=12)
ax.axhline(0, color='black', linewidth=0.8)
ax.legend(fontsize=15, loc='upper left')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()




ax.text(0.99, 0.99, f"Avg. % Change: {avg_percentual_change:.2f}%",
        ha='right', va='top', fontsize=12, color='dimgray',
        transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))



# estimation windows
window_sizes = [40,60,80]
weights_by_window = {}

# weights per window
for window in window_sizes:
    recent_returns = excess_returns.iloc[-window:]
    mu_win = excess_returns.mean().values
    Sigma_win = recent_returns.cov().values
    weights_by_window[f"{window} months"] = compute_mvo_weights(mu_win, Sigma_win)

# results to df
df_window_compare = pd.DataFrame(weights_by_window, index=assets)

df_window_compare.plot(kind='bar', figsize=(14, 6), width=0.8)
plt.title("MVO Portfolio Weights Across Different Estimation Window Lengths")
plt.ylabel("Portfolio Weight")
plt.xlabel("Assets")
plt.axhline(0, color='black', linewidth=0.5)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# These results highlight that classical MVO is not robust to sampling variation. Because it treats estimated inputs as if they are the true parameters, it reacts too aggressively to small shifts in estimated means and covariances — even when those shifts arise purely from a different sample of historical data. This makes the resulting portfolios fragile, hard to trust, and likely to perform poorly out-of-sample.

# Efficient Frontier (Windowed)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


def compute_efficient_frontier(data, bayesian=False):
    mu = data.mean().values
    S = data.cov().values
    T, N = data.shape

    if bayesian:
        # Apply correct predictive scaling
        if T <= N + 2:
            raise ValueError("Sample size T must be greater than N + 2 for Bayesian predictive covariance.")
        scale_factor = (1 + 1 / T) * ((T - 1) / (T - N - 2))
        Sigma = scale_factor * S
    else:
        Sigma = S

    target_returns = np.linspace(mu.min(), mu.max(), 100)
    portfolio_vols = []

    for target_return in target_returns:
        w = cp.Variable(N)
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)
        prob = cp.Problem(cp.Minimize(portfolio_variance),
                          [cp.sum(w) == 1,
                           portfolio_return == target_return])
        prob.solve()

        if w.value is not None:
            portfolio_vols.append(np.sqrt(portfolio_variance.value))
        else:
            portfolio_vols.append(np.nan)

    return target_returns, portfolio_vols

windows = [40,60,80]
frontiers = {}

for w in windows:
    target_returns, vols = compute_efficient_frontier(returns.iloc[:w])
    frontiers[w] = (target_returns, vols)

plt.figure(figsize=(12, 8))
for w in windows:
    plt.plot(frontiers[w][1], frontiers[w][0], label=f'Estimation Window: {w} months')

plt.xlabel('Portfolio Volatility (Std Dev)')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontiers for Different Estimation Windows')
plt.grid(True)
plt.legend()
plt.show()



def compute_efficient_frontier_general(data, bayesian=False):
    mu = data.mean().values
    S = data.cov().values
    T, N = data.shape

    if bayesian:
        if T <= N + 2:
            raise ValueError("Sample size T must be greater than N + 2 for Bayesian predictive covariance.")
        scale_factor = (1 + 1 / T) * ((T - 1) / (T - N - 2))
        Sigma = scale_factor * S
    else:
        Sigma = S

    target_returns = np.linspace(mu.min(), mu.max(), 100)
    portfolio_vols = []
    weights = []

    for target_return in target_returns:
        w = cp.Variable(N)
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)
        prob = cp.Problem(cp.Minimize(portfolio_variance),
                          [cp.sum(w) == 1, portfolio_return == target_return])
        prob.solve()
        if w.value is not None:
            portfolio_vols.append(np.sqrt(portfolio_variance.value))
            weights.append(w.value)
        else:
            portfolio_vols.append(np.nan)
            weights.append([np.nan]*N)
    return np.array(target_returns), np.array(portfolio_vols), np.array(weights)

# compute frontier
target_returns, portfolio_vols, weights = compute_efficient_frontier_general(returns.iloc[:180])

# gmv portfolio
gmv_idx = np.nanargmin(portfolio_vols)
gmv_ret = target_returns[gmv_idx]
gmv_vol = portfolio_vols[gmv_idx]
print('gmv_portfolio', gmv_ret, gmv_vol)

# tangency portfolio (max sharpe)
rf = 0.0  # rf for excess returns
sharpe = (target_returns - rf) / portfolio_vols
tangency_idx = np.nanargmax(sharpe)
tangency_ret = target_returns[tangency_idx]
tangency_vol = portfolio_vols[tangency_idx]
print('tangency_portfolio', tangency_ret, tangency_vol)

# CML
cml_x = np.linspace(0, portfolio_vols.max(), 100)
cml_y = rf + (tangency_ret - rf) / tangency_vol * cml_x

plt.figure(figsize=(10, 7))
plt.plot(portfolio_vols, target_returns, color='#0d47a1', linewidth=2.5, label='Efficient frontier')
plt.plot(gmv_vol, gmv_ret, 'o', markersize=12, color='#d32f2f', label='Global minimum variance portfolio')
plt.plot(tangency_vol, tangency_ret, 'o', markersize=12, color='#388e3c', label='Tangent (max Sharpe) portfolio')
plt.plot(cml_x, cml_y, linestyle='--', color='#616161', linewidth=2, label='Capital Market Line')

# annotation positions/colors
plt.annotate(
    'Efficient frontier',
    xy=(portfolio_vols[-1], target_returns[-1]),
    xytext=(portfolio_vols[-1]-0.007, target_returns[-1]-0.006),  # Move to the left and up
    arrowprops=dict(facecolor="#1952a7", arrowstyle='->'),
    fontsize=12, color='#0d47a1'
)
plt.annotate(
    'Global minimum\nvariance portfolio',
    xy=(gmv_vol, gmv_ret),
    xytext=(gmv_vol+0.005, gmv_ret-0.002),  # Move to the right and down
    arrowprops=dict(facecolor='#d32f2f', arrowstyle='->'),
    fontsize=12, color='#d32f2f'
)
plt.annotate(
    'Tangent portfolio',
    xy=(tangency_vol, tangency_ret),
    xytext=(tangency_vol-0.02, tangency_ret-0.002),  # Move right and up
    arrowprops=dict(facecolor='#388e3c', arrowstyle='->'),
    fontsize=12, color='#388e3c'
)

plt.xlabel('Portfolio Risk (Std Dev)', fontsize=13)
plt.ylabel('Portfolio Expected Return', fontsize=13)
plt.title('Efficient Frontier and Tangency Portfolio', fontsize=15)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Rolling Window Backtesting



def solve_portfolio(mu, Sigma, risk_limit, constraints_fn, ridge=False, lambda_ridge=0.1):
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    ret = np.array(mu) @ w
    constraints = constraints_fn(w, risk, risk_limit)

    # Objective: include ridge penalty if requested
    if ridge:
        ridge_penalty = lambda_ridge * cp.sum_squares(w)
        objective = cp.Maximize(ret - ridge_penalty)
    else:
        objective = cp.Maximize(ret)

    problem = cp.Problem(objective, constraints)
    problem.solve()
    return w.value if w.value is not None else None



# constraint cases
def unconstrained(w, risk, risk_limit):
    return [cp.sum(w) == 1, risk <= risk_limit**2]

def unconstrained_ridge(w, risk, risk_limit):
    return [cp.sum(w) == 1, risk <= risk_limit**2]

def long_only(w, risk, risk_limit):
    return [cp.sum(w) == 1, w >= 0, risk <= risk_limit**2]

def capped_0_5(w, risk, risk_limit):
    return [cp.sum(w) == 1, w >= 0, w <= 0.5, risk <= risk_limit**2]

def capped_0_3(w, risk, risk_limit):
    return [cp.sum(w) == 1, w >= 0, w <= 0.3, risk <= risk_limit**2]



from statsmodels.tsa.stattools import adfuller

def run_rolling_strategy(excess_returns, estimation_window, holding_period, risk_limit, constraint_fn):
    weights_over_time = []
    portfolio_returns = []
    portfolio_return_dates = []
    rebalance_dates = []
    T = len(excess_returns)
    asset_names_all = set()

    for start in range(0, T - estimation_window - holding_period + 1, holding_period):
        est_data = excess_returns.iloc[start:start + estimation_window].dropna(axis=1)


        '''

        hold_data = excess_returns.iloc[start + estimation_window:start + estimation_window + holding_period]
        hold_data = hold_data[est_data.columns]
        hold_dates = hold_data.index

        mu_hat = est_data.mean().values
        Sigma_hat = est_data.cov().values
        asset_names = est_data.columns
        asset_names_all.update(asset_names)

        if constraint_fn.__name__ == "unconstrained_ridge":
            weights = solve_portfolio(mu_hat, Sigma_hat, risk_limit, constraint_fn, ridge=True)
        else:
            weights = solve_portfolio(mu_hat, Sigma_hat, risk_limit, constraint_fn)

        if weights is not None:
            port_rets = hold_data @ weights
            portfolio_returns.extend(port_rets)
            portfolio_return_dates.extend(hold_dates)
            weights_over_time.append(pd.Series(weights, index=asset_names))
            rebalance_dates.append(hold_dates[0])

    weights_df = pd.DataFrame(weights_over_time, index=rebalance_dates).reindex(columns=sorted(asset_names_all)).fillna(0)
    returns_series = pd.Series(portfolio_returns, name="Rolling Portfolio excess_returns", index=portfolio_return_dates)
    return weights_df, returns_series




def summarize_performance(weights_df, returns_series):
    # === Rolling Metrics ===
    rolling_hhi = weights_df.apply(lambda w: np.sum(w**2), axis=1)
    rolling_weight_std = weights_df.std(axis=1)

    # === Turnover ===
    turnover_series = weights_df.diff().abs().sum(axis=1).iloc[1:]
    avg_turnover = turnover_series.mean()
    turnover_std = turnover_series.std()

    # === Returns & Sharpe ===
    cumulative_returns = (1 + returns_series).cumprod()
    total_cumulative_return = cumulative_returns.iloc[-1] - 1
    mean_return = returns_series.mean()
    annualized_return = ((1 + mean_return) ** 12 - 1)*100
    volatility = returns_series.std()
    sharpe_ratio = mean_return / volatility * np.sqrt(12)

    # === Summary Table ===
    summary = pd.DataFrame({
        "Total Cumulative Return": [total_cumulative_return],
        "Mean Monthly Return": [mean_return],
        "Volatility (Monthly Std)": [volatility],
        "Annualized Sharpe Ratio": [sharpe_ratio],
        "Annualized Return %": [annualized_return],
        "Average HHI": [rolling_hhi.mean()],
        "HHI Std Dev": [rolling_hhi.std()],
        "Avg Rolling Weight Std Dev": [rolling_weight_std.mean()],
        "Average Turnover": [avg_turnover],
        "Turnover Std Dev": [turnover_std]
    })

    # === Allocation Diagnostics ===
    high_weight_counts = (weights_df > 0.9).sum()
    weight_std = weights_df.std()
    avg_weight = weights_df.mean()

    allocation_diagnostics = pd.DataFrame({
        "Average Weight": avg_weight,
        "High Weight Count (>90%)": high_weight_counts,
        "Weight Std Dev": weight_std
    })

    return summary, allocation_diagnostics, cumulative_returns, turnover_series


# Strategies



strategies = {
    "Unconstrained": unconstrained,
    "Unconstrained Ridge": unconstrained_ridge,
    "Long Only": long_only,
#    "Cap 0.5": capped_0_5,
#    "Cap 0.3": capped_0_3
}

results_class = {}
for name, constraint_fn in strategies.items():

    print(f"Running strategy: {name}")
    weights_df, returns_series = run_rolling_strategy(excess_returns, 60, 2, 0.03, constraint_fn)
    # Convert excess returns to total by adding risk-free return
    rf_aligned = rf_final_aligned.reindex(returns_series.index, method='ffill')['rf_return']
    returns_total = returns_series + rf_aligned  # total returns

    # Recalculate cumulative returns
    cum_total = (1 + returns_total).cumprod() * 100

    # Store updated cumulative total returns instead of excess cumulative returns
    summary, diag, _, turnover = summarize_performance(weights_df, returns_series)
    results_class[name] = (summary, diag, cum_total, weights_df)



# insert synthetic start value
cum_total_aligned = pd.concat([
    pd.Series([100], index=[synthetic_start_date]),
    cum_total
]).sort_index()

benchmark_aligned = pd.concat([
    pd.Series([100], index=[synthetic_start_date]),
    benchmark_subset
]).sort_index()




benchmark_aligned.values



cum_total_aligned.values



plt.figure(figsize=(12, 6), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')  # white background

for name, (_, _, cum_total, _) in results_class.items():
    aligned = pd.concat([pd.Series([100], index=[cum_total.index[0]]), cum_total]).sort_index()
    plt.plot(aligned.index, aligned.values, label=name)

aligned_bench = pd.concat([pd.Series([100], index=[benchmark_subset.index[0]]), benchmark_subset]).sort_index()
plt.plot(
    aligned_bench.index, aligned_bench.values,
    linewidth=2.7, color='red', alpha=0.98,
    label="S&P 500", zorder=3
)

plt.title("Cumulative Portfolio Value of All Classic Strategies vs Benchmark (Starting from $100)", fontsize=14, weight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Portfolio Value", fontsize=12)
plt.grid(True)
plt.legend(title="Strategy")
plt.tight_layout()
plt.show()

# Strategy Comparison




summary_data = {name: result[0].iloc[0] for name, result in results_class.items()}
summary_combined = pd.DataFrame(summary_data).T

metrics_to_plot = [
    "Total Cumulative Return",
    "Mean Monthly Return",
    "Volatility (Monthly Std)",
    "Annualized Sharpe Ratio",
    "Average HHI",
    "HHI Std Dev",
    "Avg Rolling Weight Std Dev",
    "Average Turnover",
    "Turnover Std Dev",
    "Annualized Return %"
]

summary_combined_with_benchmark = pd.concat([summary_combined, benchmark_metrics])

fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(15, 5 * len(metrics_to_plot)))

for i, metric in enumerate(metrics_to_plot):
    values = summary_combined_with_benchmark[metric]
    colors = ['red' if idx == 'S&P 500' else 'skyblue' for idx in values.index]

    bars = values.plot(kind='bar', ax=axes[i], edgecolor='black', color=colors)
    axes[i].set_title(metric, fontsize=14, weight='bold')
    axes[i].set_ylabel(metric)
    axes[i].grid(axis='y', linestyle='--', linewidth=0.5)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_xlabel('')

# align x-labels right
plt.setp(axes[i].get_xticklabels(), ha="right")

plt.tight_layout()
plt.show()




# thesis metrics only
metrics_to_plot = [
    "Mean Monthly Return",
    "Volatility (Monthly Std)",
    "Annualized Sharpe Ratio",
    "Average Turnover",
    "Average HHI"
]

summary_combined_with_benchmark = pd.concat([summary_combined, benchmark_metrics])

# plot style
sns.set(style="whitegrid", font_scale=1.2, rc={"axes.facecolor": "#F8F9FB"})

fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4 * len(metrics_to_plot)))

for i, metric in enumerate(metrics_to_plot):
    values = summary_combined_with_benchmark[metric]
    colors = ['#4B8BBE' if idx != 'S&P 500' else 'red' for idx in values.index]

    bars = axes[i].bar(values.index, values, edgecolor='black', color=colors, width=0.6)

    axes[i].set_title(metric, fontsize=16, weight='bold', pad=12)
    axes[i].set_ylabel('')
    axes[i].set_xlabel('')
    axes[i].grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_alpha(0.7)
    axes[i].spines['bottom'].set_alpha(0.7)

    # Add value labels on top of bars for clarity
    for bar in bars:
        height = bar.get_height()
        axes[i].annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=11, fontweight='medium')

plt.tight_layout(h_pad=2)
plt.show()

# Strategy Comparison Insights:
# - Constrained strategies (long-only, capped) often show higher gains.
# - Unconstrained MVO can underperform in return and risk-adjusted terms.

# Bayesian Framework

# Diffuse Prior (Jeffreys)

# Bayesian Risk Perception:
# - Bayesian frontier often shifted right vs certainty-equivalence.
# - Reflects portfolio risk + estimation risk.



# Classical EF
target_returns_classic, vols_classic, weights_classic = compute_efficient_frontier_general(excess_returns, bayesian=False)

# Bayesian EF
target_returns_bayes, vols_bayes, weights_bayes = compute_efficient_frontier_general(excess_returns, bayesian=True)



def plot_efficient_frontier(target_returns, portfolio_vols, weights, label, color, dot_color):
    gmv_idx = np.nanargmin(portfolio_vols)
    gmv_ret = target_returns[gmv_idx]
    gmv_vol = portfolio_vols[gmv_idx]

    rf = 0.0  # risk-free rate for excess returns
    sharpe = (target_returns - rf) / portfolio_vols
    tangency_idx = np.nanargmax(sharpe)
    tangency_ret = target_returns[tangency_idx]
    tangency_vol = portfolio_vols[tangency_idx]

    plt.plot(portfolio_vols, target_returns, color=color, linewidth=2.5, label=label)
    plt.plot(gmv_vol, gmv_ret, 'o', markersize=10, color=dot_color, label=f'GMV ({label})')
    plt.plot(tangency_vol, tangency_ret, 'o', markersize=10, color=dot_color, label=f'Tangency ({label})')

# colors: classical blue, bayesian red
plt.figure(figsize=(10, 7))
plot_efficient_frontier(target_returns_classic, vols_classic, weights_classic, "Classical MVO", '#0d47a1', '#1976d2')
plot_efficient_frontier(target_returns_bayes, vols_bayes, weights_bayes, "Bayesian MVO (Diffuse Prior)", '#e64a19', '#ff7043')

plt.xlabel('Portfolio Risk (Std Dev)', fontsize=13)
plt.ylabel('Portfolio Expected Return', fontsize=13)
plt.title('Efficient Frontier Comparison: Classical vs Bayesian (Diffuse Prior)', fontsize=15)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# uniform axis range
plt.xlim(0, 0.055)
plt.ylim(0, 0.022)

plt.show()


# Informative Bayesian






def informative_bayes_predictive(excess_returns, eta=None, tau=None, nu=None, Omega=None,
                                  historical_mean = average_mean_historical_pre2008, historical_var=average_variance_historical_pre2008, use_historical=False):
    """
    Predictive posterior for asset returns with conjugate Bayesian prior.

    If use_sp500 is True and eta or Omega are not provided, they are set using S&P500 stats.

    Parameters:
    - excess_returns: (T x N) matrix of asset excess_returns
    - eta: prior mean vector (η), shape (N,)
    - tau: scalar confidence in eta (default = 0)
    - nu: degrees of freedom for inverse-Wishart prior (default = N + 2)
    - Omega: prior scale matrix for covariance (Ω), shape (N x N)
    - sp500_mean: scalar mean return of S&P500
    - sp500_var: scalar variance of S&P500
    - use_sp500: if True, use sp500_mean and sp500_var as default prior hyperparameters

    Returns:
    - mu_pred: predictive mean vector (length N)
    - Sigma_pred: predictive covariance matrix (N x N)
    """
    T, N = excess_returns.shape

    mu_hat = excess_returns.mean(axis=0)
    Sigma_hat = np.cov(excess_returns, rowvar=False)

    # Prior Mean
    if eta is None:
        if use_historical:
            if historical_mean is None:
                raise ValueError("sp500_mean must be provided when use_sp500 is True and eta is None")
            eta = np.full(N, historical_mean)
        else:
            eta = np.zeros(N)

    # Prior Confidence
    if tau is None:
        tau = 15  # default: sample dominates

    # Prior Degrees of Freedom
    if nu is None:
        nu = N + 2

    # Prior Covariance
    if Omega is None:
        if use_historical:
            if historical_var is None:
                raise ValueError("sp500_var must be provided when use_sp500 is True and Omega is None")
            Omega = historical_var * 1e-6 * np.eye(N)
        else:
            Omega = sample_historical_covariance * (nu - N - 1)

    # Predictive Mean
    mu_pred = (tau / (T + tau)) * eta + (T / (T + tau)) * mu_hat

    # Predictive Covariance
    diff = (np.asarray(eta) - np.asarray(mu_hat)).reshape(-1, 1)
    shrinkage_term = (tau * T) / (T + tau) * (diff @ diff.T)

    Sigma_pred = ((T + 1) / ((T+tau) * (nu + T - N - 1))) * (Omega + (T - 1) * Sigma_hat + shrinkage_term)

    return mu_pred, Sigma_pred




def compute_efficient_frontier_from_moments(mu, Sigma):
    """
    Computes the efficient frontier given mean returns and a covariance matrix.

    Parameters:
    - mu: (N,) array of expected returns
    - Sigma: (N x N) covariance matrix of returns

    Returns:
    - target_returns: array of target returns
    - portfolio_vols: array of portfolio standard deviations
    """
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)
    N = len(mu)
    target_returns = np.linspace(mu.min(), mu.max(), 100)
    portfolio_vols = []

    for target_return in target_returns:
        w = cp.Variable(N)
        port_ret = mu @ w
        port_var = cp.quad_form(w, Sigma)

        prob = cp.Problem(cp.Minimize(port_var),
                          [cp.sum(w) == 1,
                           port_ret == target_return])
        prob.solve()

        if w.value is not None:
            portfolio_vols.append(np.sqrt(port_var.value))
        else:
            portfolio_vols.append(np.nan)

    return target_returns, portfolio_vols

# Comparison of variation of weights across different estimation windows

# Efficient frontier for conjugate Bayesian MVO





# 1. Non-informative Bayesian moments
mu_noninf, Sigma_noninf = informative_bayes_predictive(
    excess_returns=excess_returns,
    use_historical=False
)

# 2. Compute SP500-based informative Bayesian predictive moments
mu_hist, Sigma_hist = informative_bayes_predictive(
    excess_returns=excess_returns,
    use_historical=True,
    tau=15
)

# 3. Compute efficient frontiers from both
ret_noninf, vol_noninf = compute_efficient_frontier_from_moments(mu_noninf, Sigma_noninf)
ret_hist, vol_hist = compute_efficient_frontier_from_moments(mu_hist, Sigma_hist)

# 4. Plot
plt.figure(figsize=(10, 6))
plt.plot(vol_noninf, ret_noninf, label="Bayesian (Non-informative)", linewidth=2)
plt.plot(vol_hist, ret_hist, label="Bayesian (History-informed prior)", linewidth=2, linestyle='--')
plt.xlabel("Portfolio Volatility (Std. Dev.)")
plt.ylabel("Expected Portfolio Return")
plt.title("Efficient Frontier: Bayesian Non-informative vs. History-Informed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





# Classical EF
target_returns_classic, vols_classic, weights_classic = compute_efficient_frontier_general(excess_returns, bayesian=False)

# Diffuse Bayesian Efficient Frontier
target_returns_bayes, vols_bayes, weights_bayes = compute_efficient_frontier_general(excess_returns, bayesian=True)

# Conjugate Bayesian (non-info prior)
mu_conj, Sigma_conj = informative_bayes_predictive(
    excess_returns,
    use_historical=False,
    tau=15,
    nu=None,
    Omega=None
)
target_returns_conj, vols_conj = compute_efficient_frontier_from_moments(mu_conj, Sigma_conj)

# Plotting Function

def plot_frontier(tr, vol, label, line_style, line_color, dot_color):
    gmv_idx = np.nanargmin(vol)
    gmv_ret = tr[gmv_idx]
    gmv_vol = vol[gmv_idx]

    sharpe = tr / vol
    tangency_idx = np.nanargmax(sharpe)
    tangency_ret = tr[tangency_idx]
    tangency_vol = vol[tangency_idx]

    plt.plot(vol, tr, linestyle=line_style, color=line_color, linewidth=2.5, label=label)
    plt.plot(gmv_vol, gmv_ret, 'o', color=dot_color, markersize=9, label=f"GMV ({label})")
    plt.plot(tangency_vol, tangency_ret, 's', color=dot_color, markersize=9, label=f"Tangency ({label})")

# Final Plot

plt.figure(figsize=(10, 7))

plot_frontier(target_returns_classic, vols_classic, "Classical MVO", '-', '#0d47a1', '#1976d2')
plot_frontier(target_returns_bayes, vols_bayes, "Bayesian MVO (Diffuse Prior)", '--', '#e64a19', '#ff7043')
plot_frontier(target_returns_conj, vols_conj, "Bayesian MVO (Conjugate Non-informative)", ':', "#08950F", "#08950F")

plt.xlabel("Portfolio Risk (Std Dev)", fontsize=13)
plt.ylabel("Portfolio Expected Return", fontsize=13)
plt.title("Efficient Frontier Comparison: Classical vs Diffuse vs Conjugate Bayesian", fontsize=15)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.055)
plt.ylim(0, 0.022)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))

# Plot each efficient frontier
plt.plot(vols_classic, target_returns_classic, label="Classical MVO", linewidth=2)
plt.plot(vols_bayes, target_returns_bayes, label="Bayesian Diffuse Prior", linestyle=':', linewidth=2)
plt.plot(vol_noninf, ret_noninf, label="Bayesian Conjugate (Non-informative)", linestyle='--', linewidth=2)
plt.plot(vol_hist, ret_hist, label="Bayesian History-Informed", linestyle='-.', linewidth=2)

plt.xlabel("Portfolio Volatility (Std. Dev.)")
plt.ylabel("Expected Portfolio Return")
plt.title("Efficient Frontiers: Classical vs Bayesian Approaches")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bayesian Rolling Window



def bayesian_diffuse_prior(est_data):
    T, N = est_data.shape
    mu_hat = est_data.mean().values
    S = est_data.cov().values
    scale_factor = (1 + 1 / T) * ((T - 1) / (T - N - 2))
    Sigma = scale_factor * S
    return mu_hat, Sigma

def bayesian_conjugate_prior(est_data):
    return informative_bayes_predictive(est_data)

def bayesian_hist_prior(est_data):
    return informative_bayes_predictive(est_data,
                                        eta=None,
                                        tau=10,
                                        nu=None,
                                        Omega=None,
                                        use_historical=True)



def run_bayesian_strategy(excess_returns, estimation_window, holding_period, risk_limit, constraint_fn, posterior_fn, ridge = False, **posterior_kwargs):
    weights_over_time = []
    portfolio_returns = []
    portfolio_return_dates = []
    rebalance_dates = []
    T = len(excess_returns)
    asset_names_all = set()

    for start in range(0, T - estimation_window - holding_period + 1, holding_period):
        est_data = excess_returns.iloc[start:start + estimation_window].dropna(axis=1)
        if est_data.shape[1] == 0:
            continue
        hold_data = excess_returns.iloc[start + estimation_window:start + estimation_window + holding_period]
        hold_data = hold_data[est_data.columns]
        hold_dates = hold_data.index

        mu_post, Sigma_post = posterior_fn(est_data, **posterior_kwargs)
        asset_names = est_data.columns
        asset_names_all.update(asset_names)

        weights = solve_portfolio(mu_post, Sigma_post, risk_limit, constraint_fn, ridge = ridge)
        if weights is None:
            print(f"[start={start}] Optimization failed for assets: {est_data.columns.tolist()}")
            continue
        if weights is not None:
            port_rets = hold_data @ weights
            portfolio_returns.extend(port_rets)
            portfolio_return_dates.extend(hold_dates)
            weights_over_time.append(pd.Series(weights, index=asset_names))
            rebalance_dates.append(hold_dates[0])

    weights_df = pd.DataFrame(weights_over_time, index=rebalance_dates).reindex(columns=sorted(asset_names_all)).fillna(0)
    returns_series = pd.Series(portfolio_returns, name="Bayesian Portfolio excess_returns", index=portfolio_return_dates)
    return weights_df, returns_series




strategies = {
    "Bayesian Diffuse": (bayesian_diffuse_prior, {}),
#    "Bayesian Historical Conjugate Prior": (bayesian_hist_prior, {}),
    "Bayesian Non-informative Conjugate Prior": (bayesian_conjugate_prior, {})
}

results_bayes = {}
for name, (posterior_fn, kwargs) in strategies.items():
    print(f"\n\n\nRunning strategy: {name}")

    weights_df, returns_series = run_bayesian_strategy(
        excess_returns, estimation_window=60, holding_period=2,
        risk_limit=0.03, constraint_fn=unconstrained,
        posterior_fn=posterior_fn, **kwargs
    )

    # Convert excess returns to total by adding risk-free return
    rf_aligned = rf_final_aligned.reindex(returns_series.index, method='ffill')['rf_return']
    returns_total = returns_series + rf_aligned  # total returns

    # Recalculate cumulative returns
    cum_total = (1 + returns_total).cumprod() * 100

    # Store updated cumulative total returns instead of excess cumulative returns
    summary, diag, _, turnover = summarize_performance(weights_df, returns_series)
    results_bayes[name] = (summary, diag, cum_total, weights_df)





strategies = {
    "Bayesian Diffuse_ridge": (bayesian_diffuse_prior, {}),
#    "Bayesian Historical Conjugate Prior": (bayesian_hist_prior, {}),
    "Bayesian Non-informative Conjugate Prior_ridge": (bayesian_conjugate_prior, {})
}

results_bayes_ridge = {}
for name, (posterior_fn, kwargs) in strategies.items():
    print(f"\n\n\nRunning strategy: {name}")

    weights_df, returns_series = run_bayesian_strategy(
        excess_returns, estimation_window=60, holding_period=2,
        risk_limit=0.03, constraint_fn=unconstrained_ridge,  # <- fixed or replace if testing constraints
        posterior_fn=posterior_fn, ridge = True ,**kwargs
    )

    # Convert excess returns to total by adding risk-free return
    rf_aligned = rf_final_aligned.reindex(returns_series.index, method='ffill')['rf_return']
    returns_total = returns_series + rf_aligned  # total returns

    # Recalculate cumulative returns
    cum_total = (1 + returns_total).cumprod() * 100

    # Store updated cumulative total returns instead of excess cumulative returns
    summary, diag, _, turnover = summarize_performance(weights_df, returns_series)
    results_bayes_ridge[name] = (summary, diag, cum_total, weights_df)



plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Portfolio Value", fontsize=12)
plt.grid(True)
plt.legend(title="Strategy")
plt.tight_layout()
plt.show()




all_results = {
    "Unconstrained": results_class["Unconstrained"],
    "Unconstrained Ridge": results_class["Unconstrained Ridge"],
    **results_bayes,
    **results_bayes_ridge
}

# Unified Cumulative Return Plot
plt.figure(figsize=(12, 6))
for name, (_, _, cum_total, _) in all_results.items():
    plt.plot(cum_total.index, cum_total.values, label=name)

plt.title("Cumulative Portfolio Value: Unconstrained Ridge vs Bayesian Strategies", fontsize=14, weight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Return", fontsize=12)
plt.legend(title="Strategy")
plt.grid(True)
plt.tight_layout()
plt.show()



# Extract Unconstrained (classical)
unconstrained_summary = {"Unconstrained": results_class["Unconstrained"][0].iloc[0]}
unconstrained_ridge_summary = {"Unconstrained Ridge": results_class["Unconstrained Ridge"][0].iloc[0]}

# Collect Bayesian summaries
summary_baye_ridge = {name: result[0].iloc[0] for name, result in results_bayes_ridge.items()}

# Collect Bayesian summaries
summary_bayes = {name: result[0].iloc[0] for name, result in results_bayes.items()}

# Combine
summary_combined = pd.DataFrame({**unconstrained_summary, **unconstrained_ridge_summary,  **summary_baye_ridge, **summary_bayes, "S&P 500": benchmark_metrics.iloc[0]}).T


# Metrics to plot
metrics_to_plot = [
    "Total Cumulative Return",
    "Mean Monthly Return",
    "Volatility (Monthly Std)",
    "Annualized Sharpe Ratio",
    "Average HHI",
    "HHI Std Dev",
    "Avg Rolling Weight Std Dev",
    "Average Turnover",
    "Turnover Std Dev",
    "Annualized Return %"
]

# Plotting
# define max/min metrics
maximize_metrics = {
    "Total Cumulative Return",
    "Mean Monthly Return",
    "Annualized Sharpe Ratio",
    "Annualized Return %"
}

fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(15, 5 * len(metrics_to_plot)))

for i, metric in enumerate(metrics_to_plot):
    values = summary_combined[metric]
    best_idx = values.idxmax() if metric in maximize_metrics else values.idxmin()
    colors = ['red' if idx == 'S&P 500' else 'orange' if idx == best_idx else 'skyblue' for idx in values.index]

    bars = values.plot(kind='bar', ax=axes[i], edgecolor='black', color=colors)
    axes[i].set_title(metric, fontsize=14, weight='bold')
    axes[i].set_ylabel(metric)
    axes[i].grid(axis='y', linestyle='--', linewidth=0.5)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_xlabel('')

# align x-labels right
plt.setp(axes[i].get_xticklabels(), ha="right")

plt.tight_layout()
plt.show()



# Extract classical strategy summaries
unconstrained_summary = {"Unconstrained": results_class["Unconstrained"][0].iloc[0]}
unconstrained_ridge_summary = {"Unconstrained Ridge": results_class["Unconstrained Ridge"][0].iloc[0]}

# === Extract Bayesian strategy summaries ===
summary_bayes = {name.replace("_ridge", ""): result[0].iloc[0] for name, result in results_bayes.items()}
summary_bayes_ridge = {name: result[0].iloc[0] for name, result in results_bayes_ridge.items()}

# === Combine grouped summaries ===
summary_unregularized = pd.DataFrame({
    **unconstrained_summary,
    **summary_bayes,
    "S&P 500": benchmark_metrics.iloc[0]
}).T

summary_ridge = pd.DataFrame({
    **unconstrained_ridge_summary,
    **summary_bayes_ridge,
    "S&P 500": benchmark_metrics.iloc[0]
}).T

# Display for verification
print("=== Unregularized ===")
display(summary_unregularized.round(4))

print("\n=== Ridge Regularised ===")
display(summary_ridge.round(4))

# Bayesian CAPM



excess_returns



historical_excess_returns



benchmark_excess_returns



excess_sp500_pre



excess_returns.index = pd.to_datetime(excess_returns.index)
benchmark_excess_returns.index = pd.to_datetime(benchmark_excess_returns.index)



from numpy.linalg import inv

def bayesian_capm_estimation(asset_returns, benchmark_returns,
                                       a0=0.0, beta0=1.0,
                                       sigma_alpha_sq=1.0, sigma_beta_sq=1.0,
                                       nu0=1, c0_sq=0.1):
    """
    Calculates the predictive mean vector and covariance matrix for a set of assets
    using the exact joint-conjugate Bayesian CAPM model described in the thesis text.

    This function aligns precisely with the theoretical specification:
    - It uses a joint conjugate Normal-Inverse-Chi-Squared prior for the
      regression parameters (alpha_i, beta_i) and idiosyncratic variance (sigma_e_i^2).
    - It uses a non-informative Jeffrey's prior for the benchmark parameters.
    - It calculates the posterior and predictive moments using the exact formulas
      derived in the text.

    Args:
        asset_returns (pd.DataFrame): DataFrame where each column is the excess return series for an asset.
                                      Must have a DateTimeIndex or similar index for alignment.
        benchmark_returns (pd.Series or pd.DataFrame): Series or single-column DataFrame of excess returns
                                                       for the benchmark. Must have the same index type as asset_returns.
        a0 (float): Prior mean for the mispricing component, alpha. Corresponds to a_0. Default is 0.
        beta0 (float): Prior mean for the market beta. Corresponds to beta_0. Default is 1.
        sigma_alpha_sq (float): Prior variance for alpha. This is interpreted as the diagonal
                                element of the prior matrix Omega_0, not including the sigma_e^2 term.
        sigma_beta_sq (float): Prior variance for beta. This is the other diagonal element of Omega_0.
        nu0 (float): Prior degrees of freedom for the idiosyncratic variance. Corresponds to nu_0.
        c0_sq (float): Prior scale parameter for the idiosyncratic variance. Corresponds to c_0^2.

    Returns:
        tuple: A tuple containing:
            - pred_mean_vector (np.array): The predictive mean vector of asset returns (mu_pred).
            - pred_cov_matrix (np.array): The predictive covariance matrix of asset returns (Sigma_pred).
    """
    # --- Data Preprocessing ---
    # filter NaNs for consistent time periods
    filtered_assets = asset_returns.dropna()
    aligned_benchmark = benchmark_returns.reindex(filtered_assets.index)

    R = filtered_assets.values
    S = aligned_benchmark.values.flatten()

    T, N = R.shape

    if T <= 3:
        raise ValueError(f"Number of observations (T={T}) must be greater than 3 for predictive moments to be defined.")
    if nu0 + T <= 2:
        raise ValueError(f"Sum of prior degrees of freedom (nu0={nu0}) and T={T} must be > 2 for posterior expectations to be defined.")

    # Stage 1: Benchmark Predictive Moments (Jeffreys)
    S_bar = np.mean(S)
    SS_S = np.sum((S - S_bar)**2)

    # Predictive var S_{t+1}
    # (Student's t for predictive density)
    var_S_pred = ((T + 1) / (T * (T - 3))) * SS_S

    # Stage 2: Asset Parameter Estimation

    # Priors (Omega_0^-1 defined)
    # We follow the standard interpretation where the prior covariance for b_i is

    b0 = np.array([a0, beta0])
    Omega0_inv = np.diag([1/sigma_alpha_sq, 1/sigma_beta_sq])

    # Design Matrix
    X = np.column_stack([np.ones(T), S])
    XTX = X.T @ X

    pred_means = np.zeros(N)
    pred_variances = np.zeros(N)
    beta_stars = np.zeros(N) # store post. mean betas (for cov calc)

    for i in range(N):
        R_i = R[:, i]

        # Posterior for Asset i

        # OLS estimate (MLE)
        b_hat_i = inv(XTX) @ X.T @ R_i

        # Post. precision/cov for b_i
        Omega_i_star_inv = Omega0_inv + XTX
        Omega_i_star = inv(Omega_i_star_inv)

        # Post. mean b_i
        b_i_star = Omega_i_star @ (Omega0_inv @ b0 + XTX @ b_hat_i)
        alpha_i_star, beta_i_star = b_i_star
        beta_stars[i] = beta_i_star

        # Post. params for sigma_e_i^2
        nu_i_star = nu0 + T

        # RSS
        rss = (R_i - X @ b_hat_i).T @ (R_i - X @ b_hat_i)

        # Shrinkage term (thesis correction)
        K_i = Omega_i_star
        shrinkage = (b0 - b_hat_i).T @ K_i @ (b0 - b_hat_i)

        # Post. scale c_i*^2
        c_i_star_sq = (1/nu_i_star) * (nu0 * c0_sq + rss + shrinkage)

        # E[sigma_e_i^2 | Data]
        exp_sigma_sq_i = (nu_i_star * c_i_star_sq) / (nu_i_star - 2)

        # Predictive Moments for Asset i

        # Predictive Mean R_{i,t+1}
        pred_means[i] = alpha_i_star + beta_i_star * S_bar

        # Predictive Var R_{i,t+1} (Eq. 15)
        # (Idiosyncratic, param uncertainty, benchmark uncertainty)

        x_tilde = np.array([1, S_bar])

        # Post. cov b_i
        cov_b_i_posterior = exp_sigma_sq_i * Omega_i_star

        var_from_coeffs = x_tilde.T @ cov_b_i_posterior @ x_tilde

        var_beta_i_posterior = cov_b_i_posterior[1, 1]
        exp_beta_i_sq = var_beta_i_posterior + beta_i_star**2

        var_from_market = exp_beta_i_sq * var_S_pred

        pred_variances[i] = exp_sigma_sq_i + var_from_coeffs + var_from_market

    # Final Predictive Mean     # --- Assemble Final Predictive Mean Vector and Covariance Matrix --- Cov
    pred_mean_vector = pred_means

    # diag matrix of pred. variances
    pred_cov_matrix = np.diag(pred_variances)

    # off-diagonal pred. covariances
    # Cov(R_i, R_j) formula
    for i in range(N):
        for j in range(i + 1, N):
            cov_ij = beta_stars[i] * beta_stars[j] * var_S_pred
            pred_cov_matrix[i, j] = cov_ij
            pred_cov_matrix[j, i] = cov_ij

    return pred_mean_vector, pred_cov_matrix

def bayesian_capm_moments(est_asset_returns, est_benchmark_returns,
                          a0=0.0, beta0=1.0, sigma_alpha_sq=1.0, sigma_beta_sq=1.0, nu0=1, c0_sq=0.1):
    """
    Wrapper to provide mean and covariance for rolling window strategies using Bayesian CAPM.
    """
    mu_pred, Sigma_pred = bayesian_capm_estimation(
        est_asset_returns, est_benchmark_returns,
        a0=a0, beta0=beta0,
        sigma_alpha_sq=sigma_alpha_sq, sigma_beta_sq=sigma_beta_sq,
        nu0=nu0, c0_sq=c0_sq
    )
    return mu_pred, Sigma_pred



# 1. Bayesian CAPM inputs
mu_capm, Sigma_capm = bayesian_capm_estimation(excess_returns, benchmark_excess_returns)

# 2. Compute the efficient frontier using your existing function
target_returns_capm, vols_capm = compute_efficient_frontier_from_moments(mu_capm, Sigma_capm)

# 3. Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(vols_capm, target_returns_capm, label="Efficient Frontier (Bayesian CAPM)", linewidth=2, color='navy')
plt.xlabel("Portfolio Volatility (Std. Dev.)")
plt.ylabel("Expected Portfolio Return")
plt.title("Efficient Frontier using Bayesian CAPM Predictive Inputs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



def compute_gmv_vol(mu, Sigma):
    import numpy as np
    import cvxpy as cp
    N = len(mu)
    w = cp.Variable(N)
    risk = cp.quad_form(w, Sigma)
    prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1])
    prob.solve()
    if w.value is not None:
        min_vol = np.sqrt(risk.value)
        return float(min_vol)
    else:
        return np.nan  # Or raise, or handle as needed



# MVO with Bayesian CAPM estimates

def capm_mvo_posterior(est_asset_returns, est_benchmark_returns):
    mu_capm, Sigma_capm = bayesian_capm_estimation(
        est_asset_returns, est_benchmark_returns
        # optional: tweak priors
    )
    return mu_capm, Sigma_capm



# Conjugate w/ Bayesian CAPM prior

def bayes_conjugate_capm_prior(est_asset_returns, est_benchmark_returns):
    N = est_asset_returns.shape[1]
    v0 = N + 10  # more informative prior
    tau = 45     # higher confidence in CAPM mean
    mu_capm, Sigma_capm = bayesian_capm_estimation(est_asset_returns, est_benchmark_returns)
    eta = mu_capm
    Omega = Sigma_capm * (v0 - N - 1)
    mu_bayes, Sigma_bayes = informative_bayes_predictive(
        est_asset_returns, eta=eta, tau=tau, nu=v0, Omega=Omega
    )
    return mu_bayes, Sigma_bayes



def run_custom_strategy(
    excess_returns,
    benchmark_returns,
    estimation_window=60,
    holding_period=2,
    risk_limit=0.03,
    constraint_fn=None,            # constraint_fn
    posterior_fn=None,             # posterior_fn (mu, Sigma)
    ridge=False,
    lambda_ridge=0.1,
    adaptive_risk=False
):
    weights_over_time = []
    portfolio_returns = []
    portfolio_return_dates = []
    rebalance_dates = []
    T = len(excess_returns)
    asset_names_all = set()

    for start in range(0, T - estimation_window - holding_period + 1, holding_period):
        est_data = excess_returns.iloc[start:start+estimation_window].dropna(axis=1)
        bench_window = benchmark_returns.iloc[start:start+estimation_window]
        if est_data.shape[1] == 0:
            continue
        hold_data = excess_returns.iloc[start+estimation_window:start+estimation_window+holding_period]
        hold_data = hold_data[est_data.columns]
        hold_dates = hold_data.index

        # Portfolio Moments (user func)
        mu, Sigma = posterior_fn(est_data, bench_window)

        # adaptive risk (Bayesian CAPM only)
        if adaptive_risk:
            gmv_vol = compute_gmv_vol(mu, Sigma)
            risk_limit_window = max(0.03, gmv_vol)
        else:
            risk_limit_window = risk_limit

        weights = solve_portfolio(
            mu, Sigma, risk_limit_window, constraint_fn, ridge=ridge, lambda_ridge=lambda_ridge
        )
        if weights is not None:
            port_rets = hold_data @ weights
            portfolio_returns.extend(port_rets)
            portfolio_return_dates.extend(hold_dates)
            weights_over_time.append(pd.Series(weights, index=est_data.columns))
            rebalance_dates.append(hold_dates[0])
            asset_names_all.update(est_data.columns)

    weights_df = pd.DataFrame(weights_over_time, index=rebalance_dates).reindex(columns=sorted(asset_names_all)).fillna(0)
    returns_series = pd.Series(portfolio_returns, name="Custom Portfolio excess_returns", index=portfolio_return_dates)
    return weights_df, returns_series




capm_strategies = {
    "Naive MVO (Bayesian CAPM)": (capm_mvo_posterior, {"ridge": False}),
    "Bayesian Conjugate (CAPM Prior)": (bayes_conjugate_capm_prior, {"ridge": False}),
    "Bayesian Conjugate (CAPM Prior) Ridge": (bayes_conjugate_capm_prior, {"ridge": True})
}




results_capm = {}
for name, (posterior_fn, kwargs) in capm_strategies.items():
    print(f"\n\n\nRunning strategy: {name}")
    ridge_flag = kwargs.pop("ridge", False)  # extract ridge flag


    weights_df, returns_series = run_custom_strategy(
        excess_returns,
        benchmark_excess_returns,
        estimation_window=60,
        holding_period=2,
        risk_limit=0.03,
        constraint_fn=unconstrained,
        posterior_fn=posterior_fn,
        ridge=ridge_flag,
        adaptive_risk=True,
        **kwargs
    )

    print(weights_df.head())


    rf_aligned = rf_final_aligned.reindex(returns_series.index, method='ffill')['rf_return']
    returns_total = returns_series + rf_aligned
    cum_total = (1 + returns_total).cumprod() * 100

    summary, diag, _, turnover = summarize_performance(weights_df, returns_series)
    results_capm[name] = (summary, diag, cum_total, weights_df)

    # plot avg weights
    avg_weight = diag["Average Weight"].sort_values(ascending=False)
    plt.figure(figsize=(14, 6))
    avg_weight.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Average Portfolio Weights Per Asset - {name}", fontsize=14, weight='bold')
    plt.xlabel("Asset", fontsize=12)
    plt.ylabel("Average Weight", fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()




all_results_capm = {
    "Classical MVO Unconstrained": results_class["Unconstrained"],
    **results_capm
}

plt.figure(figsize=(12, 6))
for name, (_, _, cum_total, _) in all_results_capm.items():
    plt.plot(cum_total.index, cum_total.values, label=name)

plt.plot(
    aligned_bench.index, aligned_bench.values,
    linewidth=2.7, color='red', alpha=0.98,
    label="S&P 500", zorder=3
)

plt.title("Cumulative Portfolio Value: CAPM-based Bayesian Strategies", fontsize=14, weight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Portfolio Value", fontsize=12)
plt.grid(True)
plt.legend(title="Strategy")
plt.tight_layout()
plt.show()






unconstrained_summary = {"Unconstrained": results_class["Unconstrained"][0].iloc[0]}
unconstrained_ridge_summary = {"Unconstrained Ridge": results_class["Unconstrained Ridge"][0].iloc[0]}
summary_bayes = {name: result[0].iloc[0] for name, result in results_bayes.items()}
summary_bayes_ridge = {name: result[0].iloc[0] for name, result in results_bayes_ridge.items()}
summary_capm = {name: result[0].iloc[0] for name, result in results_capm.items()}

summary_combined_capm = pd.DataFrame({
    **unconstrained_summary,
    **summary_capm,
    "S&P 500": benchmark_metrics.iloc[0]
}).T

# === List of metrics to plot ===
metrics_to_plot = [
    "Total Cumulative Return",
    "Mean Monthly Return",
    "Volatility (Monthly Std)",
    "Annualized Sharpe Ratio",
    "Average HHI",
    "HHI Std Dev",
    "Avg Rolling Weight Std Dev",
    "Average Turnover",
    "Turnover Std Dev",
    "Annualized Return %"
]

maximize_metrics = {
    "Total Cumulative Return",
    "Mean Monthly Return",
    "Annualized Sharpe Ratio",
    "Annualized Return %"
}

# === Plotting loop (same style as before) ===
fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(15, 5 * len(metrics_to_plot)))

for i, metric in enumerate(metrics_to_plot):
    values = summary_combined_capm[metric]
    best_idx = values.idxmax() if metric in maximize_metrics else values.idxmin()
    colors = ['red' if idx == 'S&P 500' else 'orange' if idx == best_idx else 'skyblue' for idx in values.index]
    bars = values.plot(kind='bar', ax=axes[i], edgecolor='black', color=colors)
    axes[i].set_title(metric, fontsize=14, weight='bold')
    axes[i].set_ylabel(metric)
    axes[i].grid(axis='y', linestyle='--', linewidth=0.5)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_xlabel('')

plt.setp(axes[i].get_xticklabels(), ha="right")
plt.tight_layout()
plt.show()




# Extract classical strategy summaries
unconstrained_summary = {"Unconstrained": results_class["Unconstrained"][0].iloc[0]}
unconstrained_ridge_summary = {"Unconstrained Ridge": results_class["Unconstrained Ridge"][0].iloc[0]}

# === Extract Bayesian strategy summaries ===
summary_bayes = {name.replace("_ridge", ""): result[0].iloc[0] for name, result in results_bayes.items()}
summary_bayes_ridge = {name: result[0].iloc[0] for name, result in results_bayes_ridge.items()}

# === Combine grouped summaries ===
summary_unregularized = pd.DataFrame({
    **unconstrained_summary,
    **summary_bayes,
    **summary_capm,
    "S&P 500": benchmark_metrics.iloc[0]
}).T

summary_ridge = pd.DataFrame({
    **unconstrained_ridge_summary,
    **summary_bayes_ridge,
    "S&P 500": benchmark_metrics.iloc[0]
}).T

# Display for verification
print("=== Unregularized ===")
display(summary_unregularized.round(4))

print("\n=== Ridge Regularised ===")
display(summary_ridge.round(4))



# Final Summary Plot: All Cumul. Returns
final_all_results = {
    "Unconstrained": results_class["Unconstrained"][2],
    "Ridge Regularized": results_class["Unconstrained Ridge"][2],
    "Long Only": results_class["Long Only"][2],
    **{name: val[2] for name, val in results_bayes.items()},
    **{name: val[2] for name, val in results_bayes_ridge.items()},
    **{name: val[2] for name, val in results_capm.items()}
}

# Plot all strategies
plt.figure(figsize=(14, 7))
for name, series in final_all_results.items():
    plt.plot(series.index, series.values, label=name, linewidth=1.7)

# highlight S# Highlight the S&P 500 separatelyP 500
plt.plot(
    aligned_bench.index, aligned_bench.values,
    linewidth=2.7, color='red', alpha=0.98,
    label="S&P 500", zorder=3
)

plt.title("Cumulative Portfolio Value of All Strategies vs. S&P 500 Benchmark", fontsize=15, weight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Portfolio Value ($)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Strategy", fontsize=10)
plt.tight_layout()
plt.show()




