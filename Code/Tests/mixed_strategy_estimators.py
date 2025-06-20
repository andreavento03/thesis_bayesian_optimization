import numpy as np
import pandas as pd
# import utils.bayesian_estimators (adjust path if needed)
from utils import bayesian_estimators

def calculate_mixed_hist_capm_prior_moments(
    historical_excess_returns_prior_df, # DataFrame of historical excess returns (e.g., 2002-2008)
    historical_covariance_prior_df,     # DataFrame of the historical covariance matrix (from 2002-2008)
    current_estimation_asset_returns_df, # DataFrame of asset excess returns from current backtest window
    current_estimation_benchmark_returns_series, # Series of benchmark excess returns for current window
    capm_kwargs=None
):
    """
    Calculates the mixed prior mean and mixed prior covariance.
    Mean prior = 0.5 * (historical mean) + 0.5 * (CAPM predictive mean)
    Covariance prior = 0.5 * (historical covariance) + 0.5 * (CAPM predictive covariance)

    Args:
        historical_excess_returns_prior_df (pd.DataFrame): Historical excess returns.
        historical_covariance_prior_df (pd.DataFrame): Historical covariance matrix.
        current_estimation_asset_returns_df (pd.DataFrame): Current window's asset returns.
        current_estimation_benchmark_returns_series (pd.Series): Current window's benchmark returns.
        capm_kwargs (dict, optional): Keyword arguments for bayesian_capm_estimation.

    Returns:
        tuple: (mu_mixed_prior, Sigma_avg_prior) as numpy arrays.
               Returns (None, None) if critical calculations fail.
    """
    if capm_kwargs is None:
        capm_kwargs = {}

    if current_estimation_asset_returns_df.empty or current_estimation_benchmark_returns_series.empty:
        print("Warning (MixedPriorMoments): Current estimation data or benchmark data is empty.")
        return None, None

    current_asset_names = current_estimation_asset_returns_df.columns.tolist()
    if not current_asset_names:
        print("Warning (MixedPriorMoments): No asset names in current estimation data.")
        return None, None

    # 1. capm predictive moments (current assets)
    mu_capm_pred_np, sigma_capm_pred_np = bayesian_estimators.bayesian_capm_estimation(
        current_estimation_asset_returns_df, current_estimation_benchmark_returns_series, **capm_kwargs
    )

    if mu_capm_pred_np is None or sigma_capm_pred_np is None:
        print("Warning (MixedPriorMoments): CAPM estimation failed for prior components.")
        return None, None

    mu_capm_prior_series = pd.Series(mu_capm_pred_np, index=current_asset_names)
    # ensure sigma_capm_prior_df is dataframe
    sigma_capm_prior_df = pd.DataFrame(sigma_capm_pred_np, index=current_asset_names, columns=current_asset_names)

    # 2. historical mean
    if historical_excess_returns_prior_df.empty:
        print("Warning (MixedPriorMoments): Historical prior excess returns data is empty.")
        # fallback: capm mean if no hist_mean
        mu_hist_aligned_series = mu_capm_prior_series.copy()
    else:
        common_assets_mean = historical_excess_returns_prior_df.columns.intersection(current_asset_names)
        if len(common_assets_mean) == 0:
            print("Warning (MixedPriorMoments): No common assets for historical mean. Using CAPM mean as fallback.")
            mu_hist_aligned_series = mu_capm_prior_series.copy()
        else:
            mu_hist_raw = historical_excess_returns_prior_df[common_assets_mean].mean()
            mu_hist_aligned_series = pd.Series(np.nan, index=current_asset_names)
            mu_hist_aligned_series[common_assets_mean] = mu_hist_raw
            # fill missing hist_mean with capm_mean
            mu_hist_aligned_series.fillna(mu_capm_prior_series, inplace=True)

    # 3. mixed mean prior
    mu_mixed_prior_series = 0.5 * mu_hist_aligned_series + 0.5 * mu_capm_prior_series

    # 4. align and average covariance
    if historical_covariance_prior_df.empty:
        print("Warning (MixedPriorMoments): Historical prior covariance data is empty.")
        # fallback: capm cov if no hist_cov
        historical_cov_aligned_df = sigma_capm_prior_df.copy()
    else:
        # align hist_cov to current assets
        historical_cov_aligned_df = historical_covariance_prior_df.reindex(index=current_asset_names, columns=current_asset_names)

        # fill missing hist_cov elements with capm_cov values
        for col in current_asset_names:
            if col not in historical_cov_aligned_df.columns: # Should not happen with reindex
                 historical_cov_aligned_df[col] = sigma_capm_prior_df[col]
                 continue
            if historical_cov_aligned_df[col].isnull().all(): # Whole column is NaN (asset new or no overlap)
                historical_cov_aligned_df[col] = sigma_capm_prior_df[col]
            else: # Column partially NaN (some cross-covs missing)
                historical_cov_aligned_df[col].fillna(sigma_capm_prior_df[col], inplace=True)

        # Ensure all columns from current_asset_names exist, if any were entirely missing from historical_covariance_prior_df
        for col in current_asset_names:
            if col not in historical_cov_aligned_df.columns:
                 historical_cov_aligned_df[col] = sigma_capm_prior_df[col] # Add as new column from CAPM
        # Fill any remaining NaNs that might exist (e.g. in rows)
        historical_cov_aligned_df = historical_cov_aligned_df.fillna(sigma_capm_prior_df)

    # 5. average covariance prior
    # align dataframes before averaging
    h_cov_final, c_cov_final = historical_cov_aligned_df.align(sigma_capm_prior_df, join='outer', axis=None)
    # fill nans from outer join (prefer capm)
    h_cov_final = h_cov_final.fillna(c_cov_final)
    if c_cov_final is None and h_cov_final is not None: # c_cov_final might be None if sigma_capm_prior_df was empty (should not happen due to checks)
        c_cov_final = pd.DataFrame(index=h_cov_final.index, columns=h_cov_final.columns) # create empty df to fill
    c_cov_final = c_cov_final.fillna(h_cov_final)

    Sigma_avg_prior_df = 0.5 * h_cov_final + 0.5 * c_cov_final
    if Sigma_avg_prior_df.isnull().values.any():
        print("Warning (MixedPriorMoments): NaNs found in averaged covariance matrix. Fallback to CAPM covariance.")
        Sigma_avg_prior_df = sigma_capm_prior_df.copy()


    return mu_mixed_prior_series.values, Sigma_avg_prior_df.values

def bayes_niw_with_mixed_hist_capm_prior_moments(
    est_asset_returns_df,         # Positional from engine
    est_benchmark_returns_series, # Positional from engine
    *,                            # Marker for keyword-only arguments
    historical_excess_returns_prior_df_global,
    historical_covariance_prior_df_global,
    mixed_prior_confidence_tau=10.0,
    mixed_prior_dof_nu_add=10.0,
    capm_kwargs=None
):
    """
    Calculates predictive mean and covariance using NIW model with priors derived
    from a mix of historical and CAPM estimates.

    Args:
        est_asset_returns_df (pd.DataFrame): Current window asset returns (data for NIW).
        historical_excess_returns_prior_df_global (pd.DataFrame): Global historical returns for prior.
        historical_covariance_prior_df_global (pd.DataFrame): Global historical covariance for prior.
        est_benchmark_returns_series (pd.Series): Current window benchmark returns.
        mixed_prior_confidence_tau (float): Confidence (tau) for the NIW prior mean.
        mixed_prior_dof_nu_add (float): Value to add to (N+2) for NIW DoF (nu).
        capm_kwargs (dict, optional): Keyword arguments for CAPM estimation part of the prior.

    Returns:
        tuple: (mu_pred, sigma_pred) - Predictive mean and covariance from NIW.
               Returns (None, None) if prior calculation or NIW estimation fails.
    """
    if capm_kwargs is None:
        capm_kwargs = {}

    # 1. mixed prior moments
    mu_mixed_prior, Sigma_avg_prior = calculate_mixed_hist_capm_prior_moments(
        historical_excess_returns_prior_df=historical_excess_returns_prior_df_global,
        historical_covariance_prior_df=historical_covariance_prior_df_global,
        current_estimation_asset_returns_df=est_asset_returns_df,
        current_estimation_benchmark_returns_series=est_benchmark_returns_series,
        capm_kwargs=capm_kwargs
    )

    if mu_mixed_prior is None or Sigma_avg_prior is None:
        print("Warning (NIW_MixedPrior): Failed to calculate mixed prior moments.")
        # fallback: non-informative niw if mixed prior fails
        print("Warning (NIW_MixedPrior): Falling back to non-informative NIW prior for this window.")
        return bayesian_estimators.informative_bayes_predictive_niw(
            est_asset_returns_df,
            use_flat_mean_prior=True,
            tau_prior_confidence=1e-6, # Minimal confidence
            use_flat_cov_prior=True
        )

    # 2. prepare niw priors
    eta_niw_prior_mean = mu_mixed_prior

    N_assets = est_asset_returns_df.shape[1]
    nu_niw_prior_dof = float(N_assets + 2 + mixed_prior_dof_nu_add)
    if nu_niw_prior_dof <= N_assets + 1: # ensure proper iw
        print(f"Warning (NIW_MixedPrior): Adjusted nu_niw_prior_dof from {nu_niw_prior_dof} to {float(N_assets + 2)} for proper IW.")
        nu_niw_prior_dof = float(N_assets + 2)

    # omega_0: scaled sigma_avg_prior
    omega_scale_factor = nu_niw_prior_dof - N_assets - 1.0
    # ensure positive omega_scale_factor
    if omega_scale_factor <= 1e-9:
        print(f"Warning (NIW_MixedPrior): Omega scale factor ({omega_scale_factor:.4f}) is not sufficiently positive. Adjusting nu_niw_prior_dof.")
        desired_factor = 1.0
        nu_niw_prior_dof = float(N_assets + 1.0 + desired_factor)
        omega_scale_factor = nu_niw_prior_dof - N_assets - 1.0
        print(f"Adjusted nu_niw_prior_dof to {nu_niw_prior_dof:.4f}, new omega_scale_factor: {omega_scale_factor:.4f}")

    omega_niw_prior_scale_matrix = Sigma_avg_prior * omega_scale_factor # Sigma_avg_prior is already a numpy array

    # 3. niw with mixed priors
    mu_pred, sigma_pred = bayesian_estimators.informative_bayes_predictive_niw(
        est_asset_returns_df,
        eta_prior_mean=eta_niw_prior_mean,
        tau_prior_confidence=mixed_prior_confidence_tau,
        nu_prior_dof=nu_niw_prior_dof,
        omega_prior_scale_matrix=omega_niw_prior_scale_matrix,
        use_flat_mean_prior=False, # use informative mean prior
        use_flat_cov_prior=False   # use informative cov prior
    )

    if mu_pred is None or sigma_pred is None:
        print("Warning (NIW_MixedPrior): informative_bayes_predictive_niw returned None with mixed priors.")
        return None, None

    return mu_pred, sigma_pred
