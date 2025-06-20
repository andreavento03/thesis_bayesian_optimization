import numpy as np
import pandas as pd
from numpy.linalg import inv

# NIW Conjugate Prior Estimators

def informative_bayes_predictive_niw(
    excess_returns_df,
    eta_prior_mean=None,
    tau_prior_confidence=0.0, # Default: 0 confidence (sample mean dominates if eta_prior_mean is None)
    nu_prior_dof=None, # DoF for Inverse-Wishart
    omega_prior_scale_matrix=None, # Scale matrix for Inverse-Wishart
    default_prior_historical_mean=0.0, # Fallback if eta_prior_mean is None & not use_flat_mean_prior
    default_prior_historical_var=1e-6, # Fallback for Omega if omega_prior_scale_matrix is None & not use_flat_cov_prior
    use_flat_mean_prior=True, # If True & eta_prior_mean is None: eta = zeros
    use_flat_cov_prior=True # If True & omega_prior_scale_matrix is None: non-informative Omega
):
    """
    Calculates predictive mean and covariance for asset returns using a conjugate
    Normal-Inverse-Wishart (NIW) prior. Allows for flexible prior specification.

    Args:
        excess_returns_df (pd.DataFrame): (T x N) matrix of asset excess returns.
        eta_prior_mean (np.array, optional): Prior mean vector (η) of shape (N,).
        tau_prior_confidence (float, optional): Scalar confidence in eta_prior_mean (τ).
        nu_prior_dof (float, optional): Degrees of freedom for Inverse-Wishart prior (ν). Min N+2 for proper.
        omega_prior_scale_matrix (np.array, optional): Prior scale matrix for covariance (Ω) of shape (N x N).
        default_prior_historical_mean (float, optional): Fallback for eta if not provided and not flat.
        default_prior_historical_var (float, optional): Fallback for Omega diagonal if not provided and not flat.
        use_flat_mean_prior (bool, optional): If True and eta_prior_mean is None, use zero vector for prior mean.
        use_flat_cov_prior (bool, optional): If True and omega_prior_scale_matrix is None, use a minimally informative Omega.

    Returns:
        tuple: (mu_pred, sigma_pred) - Predictive mean vector and covariance matrix.
    """
    T, N = excess_returns_df.shape
    mu_sample_mean = excess_returns_df.mean(axis=0).values
    sigma_sample_cov = np.cov(excess_returns_df, rowvar=False)

    # Prior Mean Vector (eta)
    if eta_prior_mean is None:
        if use_flat_mean_prior:
            eta_prior_mean = np.zeros(N)
        else:
            eta_prior_mean = np.full(N, default_prior_historical_mean)

    eta_prior_mean = np.asarray(eta_prior_mean).flatten()


    # Prior Degrees of Freedom (nu)
    if nu_prior_dof is None:
        nu_prior_dof = float(N + 2) # Min proper prior for IW

    # Prior Scale Matrix for Covariance (Omega)
    if omega_prior_scale_matrix is None:
        if use_flat_cov_prior:
            omega_prior_scale_matrix = np.eye(N) * 1e-6 * (nu_prior_dof - N - 1) if nu_prior_dof > N + 1 else np.eye(N) * 1e-6
        else:
            omega_prior_scale_matrix = default_prior_historical_var * np.eye(N)

    # Predictive Mean (mu_pred)
    mu_pred = (tau_prior_confidence / (T + tau_prior_confidence)) * eta_prior_mean + \
              (T / (T + tau_prior_confidence)) * mu_sample_mean

    # Predictive Covariance (Sigma_pred)
    mean_diff = (eta_prior_mean - mu_sample_mean).reshape(-1, 1)
    shrinkage_term_for_cov = (tau_prior_confidence * T) / ((T + tau_prior_confidence)) * (mean_diff @ mean_diff.T)

    omega_posterior = omega_prior_scale_matrix + (T - 1) * sigma_sample_cov + shrinkage_term_for_cov

    nu_posterior_dof = nu_prior_dof + T

    if nu_posterior_dof <= N + 1:
        print("Warning: Posterior DoF for NIW is too low. Predictive covariance might be unstable.")
        sigma_pred = sigma_sample_cov * (1 + 1/(T+tau_prior_confidence)) + np.eye(N) * 1e-7
    else:
        # Sigma_pred based on E[Sigma|data] and (1 + 1/(T+tau))
        sigma_pred = (1.0 / (nu_posterior_dof - N - 1.0)) * omega_posterior * (1.0 + (1.0 / (T + tau_prior_confidence)))

    return mu_pred, sigma_pred


# Diffuse Prior Estimator (Jeffreys')

def bayesian_diffuse_prior_moments(excess_returns_df):
    """
    Calculates predictive moments using a diffuse (Jeffreys') prior.
    """
    T, N = excess_returns_df.shape
    mu_hat = excess_returns_df.mean().values
    S_hat = excess_returns_df.cov().values

    if T <= N + 2:
        print(f"Warning: T ({T}) <= N ({N}) + 2. Diffuse prior scaling factor is undefined or unstable. Using sample covariance.")
        Sigma_pred = S_hat
    else:
        scale_factor = (1 + 1/T) * ((T - 1) / (T - N - 2))
        Sigma_pred = scale_factor * S_hat

    return mu_hat, Sigma_pred


# Bayesian CAPM Estimators

def bayesian_capm_estimation(
    asset_returns_df, benchmark_returns_series,
    a0_prior_mean_alpha=0.0, beta0_prior_mean_beta=1.0,
    sigma_alpha_sq_prior_var=1.0, sigma_beta_sq_prior_var=1.0,
    nu0_prior_dof_resid=1.0, c0_sq_prior_scale_resid=0.1
):
    """
    Calculates predictive mean and covariance using Bayesian CAPM.
    """
    R_assets = asset_returns_df.values
    S_benchmark = benchmark_returns_series.values.flatten()
    T, N_assets = R_assets.shape

    if T <= 3:
        raise ValueError(f"Observations (T={T}) must be > 3 for Bayesian CAPM predictive moments.")
    if nu0_prior_dof_resid + T <= 2:
        raise ValueError(f"Sum of prior DoF for residuals ({nu0_prior_dof_resid}) and T ({T}) must be > 2.")

    s_bar_benchmark_mean = np.mean(S_benchmark)
    ss_s_benchmark_sum_sq_dev = np.sum((S_benchmark - s_bar_benchmark_mean)**2)
    var_s_pred_benchmark_variance = ((T + 1) / (T * (T - 3))) * ss_s_benchmark_sum_sq_dev

    b0_prior_coeffs = np.array([a0_prior_mean_alpha, beta0_prior_mean_beta])
    omega0_inv_prior_precision_coeffs = np.diag([1/sigma_alpha_sq_prior_var, 1/sigma_beta_sq_prior_var])

    X_design_matrix = np.column_stack([np.ones(T), S_benchmark])
    XTX_design_matrix_sq = X_design_matrix.T @ X_design_matrix

    pred_means_assets = np.zeros(N_assets)
    pred_variances_assets = np.zeros(N_assets)
    beta_stars_posterior_mean_betas = np.zeros(N_assets)

    for i in range(N_assets):
        R_i_asset = R_assets[:, i]

        b_hat_i_ols_coeffs = inv(XTX_design_matrix_sq) @ X_design_matrix.T @ R_i_asset

        omega_i_star_inv_posterior_precision_coeffs = omega0_inv_prior_precision_coeffs + XTX_design_matrix_sq
        omega_i_star_posterior_cov_coeffs = inv(omega_i_star_inv_posterior_precision_coeffs)

        b_i_star_posterior_mean_coeffs = omega_i_star_posterior_cov_coeffs @ (omega0_inv_prior_precision_coeffs @ b0_prior_coeffs + XTX_design_matrix_sq @ b_hat_i_ols_coeffs)
        alpha_i_star, beta_i_star = b_i_star_posterior_mean_coeffs
        beta_stars_posterior_mean_betas[i] = beta_i_star

        nu_i_star_posterior_dof_resid = nu0_prior_dof_resid + T
        rss_residuals_sum_sq = (R_i_asset - X_design_matrix @ b_hat_i_ols_coeffs).T @ (R_i_asset - X_design_matrix @ b_hat_i_ols_coeffs)

        K_i_shrinkage_matrix = omega_i_star_posterior_cov_coeffs
        shrinkage_term = (b0_prior_coeffs - b_hat_i_ols_coeffs).T @ K_i_shrinkage_matrix @ (b0_prior_coeffs - b_hat_i_ols_coeffs)

        c_i_star_sq_posterior_scale_resid = (1/nu_i_star_posterior_dof_resid) * (nu0_prior_dof_resid * c0_sq_prior_scale_resid + rss_residuals_sum_sq + shrinkage_term)

        exp_sigma_sq_i_resid_variance = (nu_i_star_posterior_dof_resid * c_i_star_sq_posterior_scale_resid) / (nu_i_star_posterior_dof_resid - 2)

        pred_means_assets[i] = alpha_i_star + beta_i_star * s_bar_benchmark_mean

        x_tilde_pred_regressors = np.array([1, s_bar_benchmark_mean])
        cov_b_i_posterior_coeffs = exp_sigma_sq_i_resid_variance * omega_i_star_posterior_cov_coeffs
        var_from_coeffs = x_tilde_pred_regressors.T @ cov_b_i_posterior_coeffs @ x_tilde_pred_regressors

        var_beta_i_posterior = cov_b_i_posterior_coeffs[1, 1]
        exp_beta_i_sq_posterior = var_beta_i_posterior + beta_i_star**2
        var_from_market_uncertainty = exp_beta_i_sq_posterior * var_s_pred_benchmark_variance

        pred_variances_assets[i] = exp_sigma_sq_i_resid_variance + var_from_coeffs + var_from_market_uncertainty

    pred_mean_vector_final = pred_means_assets
    pred_cov_matrix_final = np.diag(pred_variances_assets)

    for i in range(N_assets):
        for j in range(i + 1, N_assets):
            cov_ij = beta_stars_posterior_mean_betas[i] * beta_stars_posterior_mean_betas[j] * var_s_pred_benchmark_variance
            pred_cov_matrix_final[i, j] = cov_ij
            pred_cov_matrix_final[j, i] = cov_ij

    return pred_mean_vector_final, pred_cov_matrix_final

def bayesian_capm_posterior_moments(est_asset_returns_df, est_benchmark_returns_series, **kwargs):
    """Wrapper for Bayesian CAPM estimation for use in strategy runners."""
    return bayesian_capm_estimation(est_asset_returns_df, est_benchmark_returns_series, **kwargs)

def bayes_conjugate_with_capm_prior_moments(
    est_asset_returns_df, est_benchmark_returns_series,
    capm_prior_confidence_tau=10.0,
    capm_prior_dof_nu_add=10.0, # Additional DoF for NIW prior
    **capm_kwargs
):
    """
    Uses Bayesian CAPM outputs as priors for the NIW conjugate model.
    """
    N_assets = est_asset_returns_df.shape[1]

    mu_capm_prior, sigma_capm_prior = bayesian_capm_estimation(
        est_asset_returns_df, est_benchmark_returns_series, **capm_kwargs
    )

    eta_niw_prior_mean = mu_capm_prior
    nu_niw_prior_dof = float(N_assets + 2 + capm_prior_dof_nu_add) # float for precision
    if nu_niw_prior_dof <= N_assets + 1: # Ensure proper IW
        nu_niw_prior_dof = float(N_assets + 2)

    # Ensure positive omega_scale_factor
    omega_scale_factor = nu_niw_prior_dof - N_assets - 1.0
    if omega_scale_factor <= 0:
        print(f"Warning: Omega scale factor ({omega_scale_factor}) is not positive. Adjusting nu_niw_prior_dof.")
        # Adjust nu_niw_prior_dof if omega_scale_factor not positive
        nu_niw_prior_dof = float(N_assets + 1.1 + 1.0)
        omega_scale_factor = nu_niw_prior_dof - N_assets - 1.0

    omega_niw_prior_scale_matrix = sigma_capm_prior * omega_scale_factor

    mu_pred, sigma_pred = informative_bayes_predictive_niw(
        est_asset_returns_df,
        eta_prior_mean=eta_niw_prior_mean,
        tau_prior_confidence=capm_prior_confidence_tau,
        nu_prior_dof=nu_niw_prior_dof,
        omega_prior_scale_matrix=omega_niw_prior_scale_matrix,
        use_flat_mean_prior=False, # Use eta_prior_mean from CAPM
        use_flat_cov_prior=False   # Use omega_prior_scale_matrix from CAPM
    )
    return mu_pred, sigma_pred

def mixed_historical_capm_posterior_moments(
    est_asset_returns_df,
    est_benchmark_returns_series,
    historical_prior_excess_returns_df, # Hist. excess returns for prior
    capm_kwargs=None,
    historical_mean_align_method='intersection'
):
    """
    Calculates predictive moments where the mean is a mix of historical mean and CAPM mean.
    The covariance is taken from the CAPM estimation.

    Args:
        est_asset_returns_df (pd.DataFrame): (T x N) matrix of current asset excess returns.
        est_benchmark_returns_series (pd.Series): (T x 1) series of current benchmark excess returns.
        historical_prior_excess_returns_df (pd.DataFrame): (T_hist x N_hist) of historical excess returns.
                                                           Column names should be consistent if possible.
        capm_kwargs (dict, optional): Dictionary of keyword arguments to pass to
                                      bayesian_capm_estimation.
        historical_mean_align_method (str): 'intersection' to use common assets between historical
                                            and current, or 'direct' to try and use historical means
                                            directly (requires perfect column match after selection).

    Returns:
        tuple: (mu_mixed, sigma_capm) - Mixed mean vector and CAPM covariance matrix.
               Returns (None, None) if critical data is missing or alignment fails.
    """
    if capm_kwargs is None:
        capm_kwargs = {}

    if est_asset_returns_df.empty or est_benchmark_returns_series.empty:
        print("Warning (MixedHistCAPM): Estimation data or benchmark data is empty.")
        return None, None

    current_asset_names = est_asset_returns_df.columns.tolist()
    if not current_asset_names:
        print("Warning (MixedHistCAPM): No asset names in current estimation data.")
        return None, None

    # 1. CAPM mean & cov (current assets)
    # Ensure est_asset_returns_df is clean for CAPM
    mu_capm, sigma_capm = bayesian_capm_estimation(
        est_asset_returns_df, est_benchmark_returns_series, **capm_kwargs
    )

    if mu_capm is None or sigma_capm is None:
        print("Warning (MixedHistCAPM): CAPM estimation failed.")
        return None, None

    # mu_capm to Series for alignment
    mu_capm_series = pd.Series(mu_capm, index=current_asset_names)


    # 2. Historical mean
    if historical_prior_excess_returns_df.empty:
        print("Warning (MixedHistCAPM): Historical prior excess returns data is empty. Cannot compute historical mean.")
        return None, None

    # Align hist. data cols to current assets
    common_assets = historical_prior_excess_returns_df.columns.intersection(current_asset_names)

    if historical_mean_align_method == 'intersection':
        if len(common_assets) == 0:
            print("Warning (MixedHistCAPM): No common assets between historical data and current estimation window for mean calculation.")
            return None, None

        mu_hist_raw = historical_prior_excess_returns_df[common_assets].mean()
        # Align mu_hist to current_asset_names (NaNs for missing)
        mu_hist_aligned = pd.Series(np.nan, index=current_asset_names)
        mu_hist_aligned[common_assets] = mu_hist_raw

    elif historical_mean_align_method == 'direct':
        # Assumes hist_prior_df pre-filtered
        if not all(asset in historical_prior_excess_returns_df.columns for asset in current_asset_names):
            print("Warning (MixedHistCAPM - Direct): Not all current assets found in historical data for direct mean calculation.")
            # Fallback: use available direct matches
            available_hist_assets = [col for col in current_asset_names if col in historical_prior_excess_returns_df.columns]
            if not available_hist_assets:
                 print("Warning (MixedHistCAPM - Direct Fallback): No current assets found in historical for direct mean.")
                 return None, None
            mu_hist_aligned = historical_prior_excess_returns_df[available_hist_assets].mean()
            # Reindex to match current_asset_names (fill NaN)
            mu_hist_aligned = mu_hist_aligned.reindex(current_asset_names)

        else: # All current assets in hist: proceed directly
            mu_hist_aligned = historical_prior_excess_returns_df[current_asset_names].mean()
    else:
        raise ValueError(f"Unknown historical_mean_align_method: {historical_mean_align_method}")

    # For assets missing hist. mean, fill with CAPM mean (option a).
    mu_hist_filled = mu_hist_aligned.fillna(mu_capm_series)


    # 3. Mixed mean (0.5*hist + 0.5*capm)
    mu_mixed = 0.5 * mu_hist_filled + 0.5 * mu_capm_series

    # mu_mixed to numpy array
    mu_mixed_np = mu_mixed.values

    # Sigma from CAPM (already aligned)
    return mu_mixed_np, sigma_capm
