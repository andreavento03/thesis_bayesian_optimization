import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns # For potentially nicer default styles or specific plots
import scipy.stats as stats # Added for plot_return_distribution_histograms

def plot_cumulative_wealth(
    wealth_df,
    benchmark_series=None,
    title="Cumulative Wealth (Initial $100 Investment)",
    xlabel="Date",
    ylabel="Portfolio Value ($)",
    figsize=(14, 7),
    highlight_assets=None, # List of asset names to highlight
    highlight_colors=None, # Dict of asset_name: color for highlights
    output_path=None
):
    """
    Plots cumulative wealth of multiple assets and optionally a benchmark.
    wealth_df: DataFrame where each column is an asset's wealth over time.
    benchmark_series: Series for benchmark wealth over time.
    highlight_assets: List of columns in wealth_df to plot with distinct style.
    highlight_colors: Optional dictionary mapping asset names in highlight_assets to specific colors.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    default_highlight_colors = ['green', 'blue', 'purple', 'orange', 'brown']
    used_colors = set()

    # Plot non-highlighted assets first with muted style
    for i, column in enumerate(wealth_df.columns):
        if highlight_assets and column in highlight_assets:
            continue # Skip highlighted assets for now
        series = wealth_df[column].dropna()
        ax.plot(series.index, series.values, linewidth=1, alpha=0.5, color='grey', zorder=1)

    # Plot highlighted assets
    if highlight_assets:
        color_idx = 0
        for asset_name in highlight_assets:
            if asset_name not in wealth_df.columns: continue
            series = wealth_df[asset_name].dropna()
            color = 'grey' # Default if not specified
            if highlight_colors and asset_name in highlight_colors:
                color = highlight_colors[asset_name]
            elif color_idx < len(default_highlight_colors):
                color = default_highlight_colors[color_idx]
                color_idx +=1
            ax.plot(series.index, series.values, linewidth=2.2, alpha=0.96, label=asset_name, zorder=2, color=color)
            used_colors.add(color)

    # Plot benchmark if provided
    if benchmark_series is not None:
        benchmark_label = benchmark_series.name if benchmark_series.name else "Benchmark"
        # Try to pick a benchmark color not used by highlights
        bench_color = 'red'
        if bench_color in used_colors:
            alt_bench_colors = ['black', 'darkred', 'magenta']
            for c in alt_bench_colors:
                if c not in used_colors:
                    bench_color = c
                    break
        ax.plot(benchmark_series.index, benchmark_series.dropna().values,
                linewidth=3, color=bench_color, alpha=0.98,
                label=benchmark_label, zorder=3)

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax.legend(loc='upper left', fontsize='medium', frameon=True)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    # Always show after potential save, then close.
    plt.show()
    plt.close()

def plot_annualized_returns_bar(
    annualized_returns_series,
    benchmark_name="S&P 500", # Name of the benchmark in the series' index
    title="Annualized Return per Asset and Benchmark",
    xlabel="Asset",
    ylabel="Annualized Return (%)",
    figsize=(14,7),
    output_path=None
):
    """
    Plots a bar chart of annualized returns for assets and a benchmark.
    annualized_returns_series: Series with asset names (or benchmark_name) as index and annualized returns as values.
    benchmark_name: The index label for the benchmark series.
    """
    plt.figure(figsize=figsize)
    colors = ['red' if asset == benchmark_name else 'skyblue' for asset in annualized_returns_series.index]
    annualized_returns_series.plot(kind='bar', color=colors, edgecolor='black')

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    # Always show after potential save, then close.
    plt.show()
    plt.close()

def plot_efficient_frontier(
    frontiers_data, # List of tuples: (vols, rets, label, style_dict) or (vols, rets, label)
                    # style_dict: {'line_color', 'dot_color', 'line_style', 'gmv_marker', 'tan_marker'}
    title='Efficient Frontier',
    xlabel='Portfolio Volatility (Std Dev)',
    ylabel='Portfolio Expected Return',
    figsize=(10, 7),
    xlim=None, ylim=None,
    plot_cml=False, # If true, assumes the LAST frontier in frontiers_data is the one to use for CML
    risk_free_rate=0.0,
    output_path=None
):
    """
    Plots one or more efficient frontiers, with optional GMV, Tangency points, and CML.
    frontiers_data: A list. Each element can be:
        - (vols, rets, label): Basic plotting.
        - (vols, rets, label, style_dict): For custom colors/markers.
          style_dict can contain 'line_color', 'dot_color' (for GMV/Tan), 'line_style',
          'gmv_marker' (default 'o'), 'tan_marker' (default 's').
        - (vols, rets, label, gmv_point, tan_point, style_dict): Explicit GMV/Tan points.
          gmv_point = (gmv_vol, gmv_ret), tan_point = (tan_vol, tan_ret)
    """
    plt.figure(figsize=figsize)

    last_tangency_info = None

    for item in frontiers_data:
        if len(item) == 3:
            vols, rets, label = item
            style = {}
            gmv_p, tan_p = None, None
        elif len(item) == 4:
            vols, rets, label, style = item
            gmv_p, tan_p = None, None
        elif len(item) == 6:
            vols, rets, label, gmv_p, tan_p, style = item
        else:
            raise ValueError("Each item in frontiers_data must have 3, 4, or 6 elements.")

        line_color = style.get('line_color', None) # Auto-color by default
        dot_color = style.get('dot_color', line_color or 'blue') # Match line if not specified
        line_style = style.get('line_style', '-')
        gmv_marker = style.get('gmv_marker', 'o')
        tan_marker = style.get('tan_marker', 's')

        plt.plot(vols, rets, linestyle=line_style, color=line_color, linewidth=2.5, label=label)

        if gmv_p:
            gmv_vol, gmv_ret = gmv_p
        elif not np.all(np.isnan(vols)):
            gmv_idx = np.nanargmin(vols)
            gmv_vol, gmv_ret = vols[gmv_idx], rets[gmv_idx]
        else: # Cannot determine GMV
            gmv_vol, gmv_ret = np.nan, np.nan

        if not np.isnan(gmv_vol):
             plt.plot(gmv_vol, gmv_ret, marker=gmv_marker, color=dot_color, markersize=9, label=f"GMV ({label})")

        if tan_p:
            tan_vol, tan_ret = tan_p
        elif not (np.all(np.isnan(vols)) or np.all(np.isnan(rets))):
            sharpe = (rets - risk_free_rate) / vols
            # Handle cases where vols can be zero or very small leading to inf sharpe
            valid_sharpe_indices = np.isfinite(sharpe)
            if np.any(valid_sharpe_indices):
                tangency_idx = np.nanargmax(sharpe[valid_sharpe_indices])
                actual_idx = np.where(valid_sharpe_indices)[0][tangency_idx]
                tan_vol, tan_ret = vols[actual_idx], rets[actual_idx]
                last_tangency_info = (tan_vol, tan_ret)
            else: # Cannot determine Tangency
                tan_vol, tan_ret = np.nan, np.nan
        else: # Cannot determine Tangency
            tan_vol, tan_ret = np.nan, np.nan

        if not np.isnan(tan_vol):
            plt.plot(tan_vol, tan_ret, marker=tan_marker, color=dot_color, markersize=9, label=f"Tangency ({label})")

    if plot_cml and last_tangency_info:
        tan_vol, tan_ret = last_tangency_info
        if not (np.isnan(tan_vol) or tan_vol == 0): # Ensure tan_vol is not zero for CML slope
            cml_x = np.linspace(0, np.nanmax(vols) if not np.all(np.isnan(vols)) else 0.1, 100)
            cml_y = risk_free_rate + (tan_ret - risk_free_rate) / tan_vol * cml_x
            plt.plot(cml_x, cml_y, linestyle='--', color='grey', linewidth=2, label='Capital Market Line')

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    # Always show after potential save, then close.
    plt.show()
    plt.close()

def plot_average_weights_bar(
    avg_weights_series,
    strategy_name="Strategy",
    title_prefix="Average Portfolio Weights Per Asset",
    xlabel="Asset", ylabel="Average Weight",
    figsize=(14, 6),
    output_path=None
):
    """
    Plots a bar chart of average portfolio weights for a strategy.
    """
    plt.figure(figsize=figsize)
    avg_weights_series.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"{title_prefix} - {strategy_name}", fontsize=14, weight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    # Always show after potential save, then close.
    plt.show()
    plt.close()

def plot_strategy_cumulative_returns_comparison(
    cumulative_returns_dict, # {strategy_name: pd.Series(cum_returns)}
    benchmark_series=None, # pd.Series for benchmark cumulative returns, already rebased to 100
    benchmark_label="S&P 500",
    title="Cumulative Portfolio Value Comparison (Starting $100)",
    xlabel="Date", ylabel="Cumulative Portfolio Value ($)",
    figsize=(12, 7),
    output_path=None
):
    """
    Plots cumulative returns for multiple strategies against each other and an optional benchmark.
    Assumes all input series start at the same base value (e.g., 100).
    """
    plt.figure(figsize=figsize)
    for name, cum_total_series in cumulative_returns_dict.items():
        plt.plot(cum_total_series.index, cum_total_series.values, label=name, linewidth=1.7)

    if benchmark_series is not None:
        plt.plot(benchmark_series.index, benchmark_series.values,
                 linewidth=2.7, color='red', alpha=0.98,
                 label=benchmark_label, zorder=3)

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Strategy")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    # Always show after potential save, then close.
    plt.show()
    plt.close()

def plot_strategy_metrics_comparison_bar(
    summary_df, # DataFrame where index is strategy name, columns are metrics
    metrics_to_plot, # List of metric column names to plot
    benchmark_name="S&P 500",
    maximize_metrics=None, # Set of metric names that are better when maximized
    figsize_per_metric=(12, 4),
    output_path=None
):
    """
    Plots bar charts for comparing multiple performance metrics across different strategies.
    """
    if maximize_metrics is None:
        maximize_metrics = set()

    num_metrics = len(metrics_to_plot)
    plt.figure(figsize=(figsize_per_metric[0], figsize_per_metric[1] * num_metrics))

    for i, metric in enumerate(metrics_to_plot):
        ax = plt.subplot(num_metrics, 1, i + 1)
        values = summary_df[metric]

        best_val = values.max() if metric in maximize_metrics else values.min()
        colors = []
        for idx in values.index:
            if idx == benchmark_name:
                colors.append('red')
            elif values[idx] == best_val:
                colors.append('orange') # Highlight best performing strategy (excluding benchmark)
            else:
                colors.append('skyblue')

        bars = values.plot(kind='bar', ax=ax, edgecolor='black', color=colors, width=0.6)
        ax.set_title(metric, fontsize=14, weight='bold', pad=10)
        ax.set_ylabel('') # Metric name is title
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
        ax.tick_params(axis='x', labelrotation=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add value labels
        for bar in bars.patches: # Use .patches for bar objects
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout(h_pad=2.5)
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    # Always show after potential save, then close.
    plt.show()
    plt.close()

print("Code/utils/plotting_utils.py created with initial set of plotting functions.")


# --- Additional Plotting Functions ---

def plot_return_distribution_histograms(
    returns_data_df,
    log_returns_data_df=None,
    num_assets_to_sample=5,
    bins=30,
    title_prefix="Return Distributions",
    figsize_per_row=(14, 2.5), # Figsize for each row of 5 assets
    output_path_raw=None,
    output_path_log=None,
    output_path_sampled_comparison=None
):
    """
    Plots histograms of raw returns and optionally log returns,
    with Normal and KDE overlays. Also plots a comparison for a sample of assets.
    """
    # Plot all raw returns distributions
    if not returns_data_df.empty:
        num_assets_raw = returns_data_df.shape[1]
        ncols_raw = min(5, num_assets_raw) # Max 5 plots per row
        nrows_raw = int(np.ceil(num_assets_raw / ncols_raw))
        fig_raw, axes_raw = plt.subplots(nrows=nrows_raw, ncols=ncols_raw,
                                         figsize=(ncols_raw * 2.8, nrows_raw * 2.5), squeeze=False)
        axes_raw = axes_raw.flatten()
        for i, col in enumerate(returns_data_df.columns):
            data = returns_data_df[col].dropna()
            if data.empty: continue
            ax = axes_raw[i]
            ax.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Actual')
            mu, std = data.mean(), data.std()
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            ax.plot(x, stats.norm.pdf(x, mu, std), 'k-', linewidth=1.5, label='Normal')
            try:
                kde = stats.gaussian_kde(data)
                ax.plot(x, kde(x), 'r--', linewidth=1.5, label='KDE')
            except np.linalg.LinAlgError: # if singular matrix for KDE
                print(f"Could not compute KDE for {col}")
            ax.set_title(col, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0: ax.legend(fontsize=7)
        for j in range(i + 1, len(axes_raw)):
            fig_raw.delaxes(axes_raw[j])
        fig_raw.suptitle(f"{title_prefix} - Raw Returns", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if output_path_raw:
            plt.savefig(output_path_raw)
            print(f"Raw returns distribution plot saved to {output_path_raw}")
        plt.show() # Show the specific figure
        plt.close(fig_raw) # Close the specific figure

    # Plot all log returns distributions (if provided)
    if log_returns_data_df is not None and not log_returns_data_df.empty:
        num_assets_log = log_returns_data_df.shape[1]
        ncols_log = min(5, num_assets_log)
        nrows_log = int(np.ceil(num_assets_log / ncols_log))
        fig_log, axes_log = plt.subplots(nrows=nrows_log, ncols=ncols_log,
                                         figsize=(ncols_log * 2.8, nrows_log * 2.5), squeeze=False)
        axes_log = axes_log.flatten()
        for i, col in enumerate(log_returns_data_df.columns):
            data = log_returns_data_df[col].dropna()
            if data.empty: continue
            ax = axes_log[i]
            ax.hist(data, bins=bins, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Actual')
            mu, std = data.mean(), data.std()
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            ax.plot(x, stats.norm.pdf(x, mu, std), 'k-', linewidth=1.5, label='Normal')
            try:
                kde = stats.gaussian_kde(data)
                ax.plot(x, kde(x), 'r--', linewidth=1.5, label='KDE')
            except np.linalg.LinAlgError:
                print(f"Could not compute KDE for {col} (log returns)")
            ax.set_title(col, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0: ax.legend(fontsize=7)
        for j in range(i + 1, len(axes_log)):
            fig_log.delaxes(axes_log[j])
        fig_log.suptitle(f"{title_prefix} - Log Returns", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if output_path_log:
            plt.savefig(output_path_log)
            print(f"Log returns distribution plot saved to {output_path_log}")
        plt.show() # Show the specific figure
        plt.close(fig_log) # Close the specific figure

    # Plot comparison for a sample of assets (Raw vs Log, similar to original cell In[45])
    if not returns_data_df.empty and log_returns_data_df is not None and not log_returns_data_df.empty and num_assets_to_sample > 0:
        np.random.seed(42) # for reproducibility of sample
        sampled_cols = np.random.choice(returns_data_df.columns,
                                        size=min(num_assets_to_sample, len(returns_data_df.columns)),
                                        replace=False)

        fig_samp, axes_samp = plt.subplots(nrows=2, ncols=len(sampled_cols),
                                           figsize=(len(sampled_cols) * 2.8, 5), squeeze=False)

        for i, col in enumerate(sampled_cols):
            # Raw returns
            data_raw = returns_data_df[col].dropna()
            ax_raw = axes_samp[0, i]
            if not data_raw.empty:
                ax_raw.hist(data_raw, bins=bins, density=True, alpha=0.5, color='steelblue', edgecolor='black')
                mu_r, std_r = data_raw.mean(), data_raw.std()
                xr = np.linspace(*ax_raw.get_xlim(), 100)
                ax_raw.plot(xr, stats.norm.pdf(xr, mu_r, std_r), 'k-', linewidth=1.3, label='Normal')
                try: ax_raw.plot(xr, stats.gaussian_kde(data_raw)(xr), 'r--', linewidth=1.3, label='KDE')
                except np.linalg.LinAlgError: pass
            ax_raw.set_title(f"Raw: {col}", fontsize=10)
            ax_raw.set_xticks([]); ax_raw.set_yticks([])
            if i == 0: ax_raw.legend(fontsize=7, loc='upper right')

            # Log returns
            if col in log_returns_data_df.columns:
                data_log = log_returns_data_df[col].dropna()
                ax_log = axes_samp[1, i]
                if not data_log.empty:
                    ax_log.hist(data_log, bins=bins, density=True, alpha=0.5, color='seagreen', edgecolor='black')
                    mu_l, std_l = data_log.mean(), data_log.std()
                    xl = np.linspace(*ax_log.get_xlim(), 100)
                    ax_log.plot(xl, stats.norm.pdf(xl, mu_l, std_l), 'k-', linewidth=1.3)
                    try: ax_log.plot(xl, stats.gaussian_kde(data_log)(xl), 'r--', linewidth=1.3)
                    except np.linalg.LinAlgError: pass
                ax_log.set_title(f"Log: {col}", fontsize=10)
                ax_log.set_xticks([]); ax_log.set_yticks([])

        fig_samp.suptitle(f"{title_prefix} - Sampled Raw vs. Log Distributions", fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        if output_path_sampled_comparison:
            plt.savefig(output_path_sampled_comparison)
            print(f"Sampled comparison plot saved to {output_path_sampled_comparison}")
        plt.show() # Show the specific figure
        plt.close(fig_samp) # Close the specific figure

def plot_weights_comparison_bar(
    weights_df_dict, # Dict: {'Scenario A': weights_series_A, 'Scenario B': weights_series_B}
    title="Portfolio Weights Comparison",
    xlabel="Assets",
    ylabel="Portfolio Weight",
    figsize=(14, 7),
    output_path=None
):
    """
    Plots a bar chart comparing different sets of portfolio weights for the same assets.
    weights_df_dict: Dictionary where keys are scenario names and values are Series/DataFrames of weights.
                     If values are DataFrames, it plots the first series/column or means if appropriate.
                     For simplicity, assumes values are Series with asset names as index.
    """
    if not isinstance(weights_df_dict, dict) or not weights_df_dict:
        print("weights_df_dict must be a non-empty dictionary.")
        return

    # Convert dict of Series to DataFrame for easier plotting
    plot_data = pd.DataFrame(weights_df_dict)

    plot_data.plot(kind='bar', figsize=figsize, width=0.8, alpha=0.9, edgecolor='k')

    plt.title(title, fontsize=15, pad=10)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend(title="Scenario", fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    # Always show after potential save, then close.
    plt.show()
    plt.close()

print("Additional plotting functions appended to Code/utils/plotting_utils.py")
