import numpy as np
import pandas as pd
import cvxpy as cp
from numpy.linalg import inv

# --- MVO Weight Calculation Functions ---

def compute_mvo_weights(mu_values, sigma_matrix):
    """
    Computes MVO weights for the tangency portfolio (unconstrained, maximizes Sharpe assuming rf=0).
    mu_values: array of mean returns
    sigma_matrix: covariance matrix
    """
    inv_sigma = inv(sigma_matrix)
    weights = inv_sigma @ mu_values
    weights /= np.sum(weights) # Normalize to sum to 1
    return weights

def compute_long_only_mvo_weights(mu_values, sigma_matrix, risk_limit_sq=None):
    """
    Computes long-only MVO weights, maximizing return for a given risk limit.
    If risk_limit_sq is None, it might not behave as expected or may need reformulation
    to maximize Sharpe or another objective. The original notebook's specific formulation:
    problem = cp.Problem(cp.Maximize(ret), constraints) implies maximizing return.

    mu_values: array of mean returns
    sigma_matrix: covariance matrix
    risk_limit_sq: squared risk limit (volatility squared)
    """
    n = len(mu_values)
    w = cp.Variable(n)

    portfolio_return = mu_values @ w
    portfolio_risk = cp.quad_form(w, sigma_matrix)

    constraints = [cp.sum(w) == 1, w >= 0]
    if risk_limit_sq is not None:
        constraints.append(portfolio_risk <= risk_limit_sq)

    problem = cp.Problem(cp.Maximize(portfolio_return), constraints)
    problem.solve()

    if w.value is not None:
        return w.value
    else:
        # Fallback or error handling if optimization fails
        print("Warning: Long-only MVO optimization failed. Returning equal weights.")
        return np.full(n, 1/n)


# --- Core Portfolio Solver ---

def solve_portfolio(mu_values, sigma_matrix, risk_limit_sq, constraints_fn, ridge=False, lambda_ridge=0.1):
    """
    Solves the portfolio optimization problem with given constraints and optional ridge regularization.
    risk_limit_sq: The risk limit (volatility SQUARED).
    """
    n = len(mu_values)
    w = cp.Variable(n)

    # Ensure mu_values is a 1D numpy array
    mu_np = np.array(mu_values).flatten()

    portfolio_return = mu_np @ w
    portfolio_risk = cp.quad_form(w, sigma_matrix)

    active_constraints = constraints_fn(w, portfolio_risk, risk_limit_sq)

    if ridge:
        ridge_penalty = lambda_ridge * cp.sum_squares(w)
        objective = cp.Maximize(portfolio_return - ridge_penalty)
    else:
        objective = cp.Maximize(portfolio_return)

    problem = cp.Problem(objective, active_constraints)
    problem.solve(solver=cp.ECOS) # Specifying a solver can sometimes help

    if w.value is not None:
        return w.value
    else:
        # Fallback if optimization fails (e.g., infeasible)
        # This might happen if risk_limit is too low for target returns implicit in constraints
        print(f"Warning: Optimization failed for solve_portfolio with constraints {constraints_fn.__name__}. Risk limit: {np.sqrt(risk_limit_sq):.4f}")
        # Attempt a simpler objective: minimize risk for sum(w)=1 and any other simple constraints
        try:
            print("Attempting fallback: Minimize risk with basic constraints.")
            fallback_constraints = [cp.sum(w) == 1]
            if 'long_only' in constraints_fn.__name__ or 'capped' in constraints_fn.__name__:
                fallback_constraints.append(w >= 0)

            fallback_problem = cp.Problem(cp.Minimize(portfolio_risk), fallback_constraints)
            fallback_problem.solve(solver=cp.ECOS)
            if w.value is not None:
                print("Fallback succeeded.")
                return w.value
            else:
                print("Fallback also failed. Returning equal weights.")
        except Exception as e:
            print(f"Error during fallback: {e}")

        return np.full(n, 1/n) # Default to equal weights if all else fails


# --- Constraint Functions ---
# These functions now expect risk_limit_sq (volatility squared)

def unconstrained(w_var, risk_var, risk_limit_sq_val):
    return [cp.sum(w_var) == 1, risk_var <= risk_limit_sq_val]

def unconstrained_ridge(w_var, risk_var, risk_limit_sq_val): # Same as unconstrained for solve_portfolio
    return [cp.sum(w_var) == 1, risk_var <= risk_limit_sq_val]

def long_only(w_var, risk_var, risk_limit_sq_val):
    return [cp.sum(w_var) == 1, w_var >= 0, risk_var <= risk_limit_sq_val]

def capped_0_5(w_var, risk_var, risk_limit_sq_val):
    return [cp.sum(w_var) == 1, w_var >= 0, w_var <= 0.5, risk_var <= risk_limit_sq_val]

def capped_0_3(w_var, risk_var, risk_limit_sq_val):
    return [cp.sum(w_var) == 1, w_var >= 0, w_var <= 0.3, risk_var <= risk_limit_sq_val]


# --- Efficient Frontier Functions ---

def compute_efficient_frontier_from_moments(mu_values, sigma_matrix, num_points=100):
    """
    Computes the efficient frontier given mean returns and a covariance matrix.
    """
    mu_np = np.asarray(mu_values).flatten()
    sigma_np = np.asarray(sigma_matrix)
    n = len(mu_np)

    # Determine a reasonable range for target returns
    min_ret_possible = np.min(mu_np) # Simplistic, assumes single asset holding
    max_ret_possible = np.max(mu_np) # Simplistic

    # If all mu_values are very close, linspace might create identical points
    if np.isclose(min_ret_possible, max_ret_possible):
        target_returns_np = np.array([min_ret_possible])
    else:
        target_returns_np = np.linspace(min_ret_possible, max_ret_possible, num_points)

    portfolio_vols_np = []
    all_weights = []

    for target_ret_val in target_returns_np:
        w = cp.Variable(n)
        portfolio_return_var = mu_np @ w
        portfolio_variance_var = cp.quad_form(w, sigma_np)

        constraints = [cp.sum(w) == 1, portfolio_return_var == target_ret_val]
        problem = cp.Problem(cp.Minimize(portfolio_variance_var), constraints)
        problem.solve(solver=cp.ECOS)

        if w.value is not None and portfolio_variance_var.value is not None and portfolio_variance_var.value >= 0:
            portfolio_vols_np.append(np.sqrt(portfolio_variance_var.value))
            all_weights.append(w.value)
        else:
            portfolio_vols_np.append(np.nan)
            all_weights.append(np.full(n, np.nan))

    return np.array(target_returns_np), np.array(portfolio_vols_np), np.array(all_weights)

def compute_efficient_frontier_general(returns_df, bayesian_scale_factor_fn=None, num_points=100):
    """
    Computes efficient frontier from returns data, optionally applying Bayesian scaling.
    bayesian_scale_factor_fn: A function that takes (T, N) and returns a scaling factor for Sigma.
                              Example: def diffuse_prior_scale(T, N): return (1 + 1/T) * ((T - 1) / (T - N - 2))
    """
    mu_hat = returns_df.mean().values
    s_hat = returns_df.cov().values
    t, n = returns_df.shape

    sigma_eff = s_hat
    if bayesian_scale_factor_fn:
        if t <= n + 2: # Condition for diffuse prior scaling
            print(f"Warning: T ({t}) <= N ({n}) + 2. Bayesian scaling factor for diffuse prior might be undefined or unstable. Using sample covariance.")
        else:
            scale_factor = bayesian_scale_factor_fn(t, n)
            sigma_eff = scale_factor * s_hat

    return compute_efficient_frontier_from_moments(mu_hat, sigma_eff, num_points)

def compute_gmv_vol(mu_values, sigma_matrix):
    """
    Computes the volatility of the Global Minimum Variance (GMV) portfolio.
    """
    n = len(mu_values)
    w = cp.Variable(n)
    risk = cp.quad_form(w, sigma_matrix)
    problem = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1])
    problem.solve(solver=cp.ECOS)

    if w.value is not None and risk.value is not None and risk.value >=0:
        min_vol = np.sqrt(risk.value)
        return float(min_vol)
    else:
        print("Warning: GMV computation failed.")
        return np.nan

print("Code/utils/portfolio_optimizers.py created.")
