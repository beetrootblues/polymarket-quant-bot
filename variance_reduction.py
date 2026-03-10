"""
Polymarket Quant Bot — Variance Reduction Engine (Article Part V)
==================================================================
Three techniques that "combine multiplicatively" (from the article):

1. Antithetic Variates — "Free Symmetry"
   Typical reduction: 50-75%. Zero extra computational cost.

2. Control Variates — "Exploit What You Already Know"
   Use Black-Scholes digital price as control variate.

3. Stratified Sampling — "Divide and Conquer"  
   Partition probability space, sample within each stratum.

"Stack all three: antithetic variates inside each stratum, with a
control variate correction — and you routinely achieve 100-500x
variance reduction over crude MC. This is not optional in production.
This is table stakes."
"""
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
from config import VarianceReductionConfig


# --- Technique 1: Antithetic Variates ----------------------------------------

def antithetic_binary_mc(
    S0: float,
    K: float,
    sigma: float,
    T: float,
    mu: float = 0.0,
    N_paths: int = 100_000,
) -> Dict[str, float]:
    """
    Antithetic variates for binary contract pricing.
    From Article Part V: "Free Symmetry"

    For every Z, also use -Z. When the payoff is monotone
    (binary contracts always are), variance reduction is guaranteed.

    "Typical reduction is around 50-75%. Zero extra computational cost
    beyond doubling the function evaluations (which you were going to do anyway)."
    """
    n_half = N_paths // 2
    Z = np.random.standard_normal(n_half)

    # Original paths
    S_T_pos = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs_pos = (S_T_pos > K).astype(float)

    # Antithetic paths (negate Z)
    S_T_neg = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (-Z))
    payoffs_neg = (S_T_neg > K).astype(float)

    # Antithetic estimator: average of each pair
    paired_estimates = (payoffs_pos + payoffs_neg) / 2

    p_anti = float(paired_estimates.mean())
    se_anti = float(paired_estimates.std() / np.sqrt(n_half))

    # Crude MC comparison (same total paths)
    Z_crude = np.random.standard_normal(N_paths)
    S_T_crude = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_crude)
    p_crude = float((S_T_crude > K).mean())
    se_crude = float(np.sqrt(p_crude * (1 - p_crude) / N_paths)) if 0 < p_crude < 1 else 0.0

    vr_factor = (se_crude / se_anti) ** 2 if se_anti > 0 else float("inf")

    return {
        "p_antithetic": p_anti,
        "se_antithetic": se_anti,
        "p_crude": p_crude,
        "se_crude": se_crude,
        "variance_reduction": float(vr_factor),
        "ci_95": (p_anti - 1.96 * se_anti, p_anti + 1.96 * se_anti),
    }


# --- Technique 2: Control Variates -------------------------------------------

def control_variate_binary_mc(
    S0: float,
    K: float,
    sigma: float,
    T: float,
    mu: float = 0.0,
    r: float = 0.05,
    N_paths: int = 100_000,
) -> Dict[str, float]:
    """
    Control variate MC for binary contract pricing.
    From Article Part V: "Exploit What You Already Know"

    Uses the Black-Scholes digital option price (closed form) as control:

    p_cv = p_hat - beta * (C_hat_BS - C_BS)

    where C_BS is the known BS digital price and C_hat_BS is its MC estimate.
    """
    Z = np.random.standard_normal(N_paths)
    S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Target: binary payoff under actual model
    payoffs = (S_T > K).astype(float)
    p_crude = float(payoffs.mean())

    # Control variate: Black-Scholes digital price (closed-form)
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    p_BS_exact = float(norm.cdf(d2))

    # MC estimate of BS digital under same random numbers
    S_T_bs = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs_bs = (S_T_bs > K).astype(float)
    p_BS_mc = payoffs_bs.mean()

    # Optimal beta: Cov(Y, C) / Var(C)
    cov_matrix = np.cov(payoffs, payoffs_bs)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0

    # Control variate estimator
    p_cv = float(p_crude - beta * (p_BS_mc - p_BS_exact))

    # Variance of CV estimator
    adjusted = payoffs - beta * (payoffs_bs - p_BS_exact)
    se_cv = float(adjusted.std() / np.sqrt(N_paths))
    se_crude = float(np.sqrt(p_crude * (1 - p_crude) / N_paths)) if 0 < p_crude < 1 else 0.0

    vr_factor = (se_crude / se_cv) ** 2 if se_cv > 0 else float("inf")

    return {
        "p_control_variate": p_cv,
        "se_control_variate": se_cv,
        "p_crude": p_crude,
        "se_crude": se_crude,
        "p_BS_exact": p_BS_exact,
        "beta_optimal": float(beta),
        "variance_reduction": float(vr_factor),
        "ci_95": (p_cv - 1.96 * se_cv, p_cv + 1.96 * se_cv),
    }


# --- Technique 3: Stratified Sampling ----------------------------------------

def stratified_binary_mc(
    S0: float,
    K: float,
    sigma: float,
    T: float,
    mu: float = 0.0,
    J: int = 10,
    N_total: int = 100_000,
) -> Dict[str, float]:
    """
    Stratified MC for binary contract pricing.
    Directly from Article Part V: "Divide and Conquer"

    Strata defined by quantiles of the terminal price distribution.

    "Variance is always <= crude MC (by the law of total variance),
    with maximum gain from Neyman allocation."
    """
    n_per_stratum = N_total // J
    estimates = []

    for j in range(J):
        # Uniform draws within stratum [j/J, (j+1)/J]
        U = np.random.uniform(j / J, (j + 1) / J, n_per_stratum)
        Z = norm.ppf(U)
        S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        stratum_mean = (S_T > K).mean()
        estimates.append(stratum_mean)

    # Each stratum has weight 1/J
    p_stratified = float(np.mean(estimates))
    se_stratified = float(np.std(estimates) / np.sqrt(J))

    # Crude comparison
    Z_crude = np.random.standard_normal(N_total)
    S_T_crude = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_crude)
    p_crude = float((S_T_crude > K).mean())
    se_crude = float(np.sqrt(p_crude * (1 - p_crude) / N_total)) if 0 < p_crude < 1 else 0.0

    vr_factor = (se_crude / se_stratified) ** 2 if se_stratified > 0 else float("inf")

    return {
        "p_stratified": p_stratified,
        "se_stratified": se_stratified,
        "p_crude": p_crude,
        "se_crude": se_crude,
        "n_strata": J,
        "variance_reduction": float(vr_factor),
        "ci_95": (p_stratified - 1.96 * se_stratified, p_stratified + 1.96 * se_stratified),
    }


# --- Combined: All Three Stacked ---------------------------------------------

def stacked_variance_reduction(
    S0: float,
    K: float,
    sigma: float,
    T: float,
    mu: float = 0.0,
    r: float = 0.05,
    J: int = 10,
    N_total: int = 100_000,
    config: Optional[VarianceReductionConfig] = None,
) -> Dict[str, float]:
    """
    Stack ALL THREE techniques as described in the article:

    "Antithetic variates inside each stratum, with a control variate
    correction — and you routinely achieve 100-500x variance reduction
    over crude MC."

    Architecture:
    1. Stratified sampling divides the probability space
    2. Within each stratum, use antithetic pairs
    3. Apply control variate correction globally
    """
    if config:
        J = config.n_strata

    n_per_stratum = N_total // J
    n_half = n_per_stratum // 2  # For antithetic pairs within strata

    # Black-Scholes exact price for control variate
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    p_BS_exact = float(norm.cdf(d2))

    all_payoffs = []
    all_payoffs_bs = []

    for j in range(J):
        # Stratified uniform draws
        U = np.random.uniform(j / J, (j + 1) / J, n_half)
        Z = norm.ppf(U)

        # Antithetic pairs within stratum
        Z_combined = np.concatenate([Z, -Z])

        # Target model paths
        S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_combined)
        payoffs = (S_T > K).astype(float)

        # Control variate paths (BS model)
        S_T_bs = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_combined)
        payoffs_bs = (S_T_bs > K).astype(float)

        all_payoffs.extend(payoffs)
        all_payoffs_bs.extend(payoffs_bs)

    all_payoffs = np.array(all_payoffs)
    all_payoffs_bs = np.array(all_payoffs_bs)

    # Control variate correction
    cov_matrix = np.cov(all_payoffs, all_payoffs_bs)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0

    adjusted = all_payoffs - beta * (all_payoffs_bs - p_BS_exact)
    p_stacked = float(adjusted.mean())
    se_stacked = float(adjusted.std() / np.sqrt(len(adjusted)))

    # Crude MC comparison
    Z_crude = np.random.standard_normal(N_total)
    S_T_crude = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_crude)
    p_crude = float((S_T_crude > K).mean())
    se_crude = float(np.sqrt(p_crude * (1 - p_crude) / N_total)) if 0 < p_crude < 1 else 0.0

    vr_factor = (se_crude / se_stacked) ** 2 if se_stacked > 0 else float("inf")

    return {
        "p_stacked": p_stacked,
        "se_stacked": se_stacked,
        "p_crude": p_crude,
        "se_crude": se_crude,
        "p_BS_exact": p_BS_exact,
        "beta": float(beta),
        "total_variance_reduction": float(vr_factor),
        "ci_95": (p_stacked - 1.96 * se_stacked, p_stacked + 1.96 * se_stacked),
        "technique": "stratified + antithetic + control_variate",
    }


# --- Polymarket-Specific: Probability Contract VR ----------------------------

def polymarket_probability_estimate(
    current_price: float,
    vol_estimate: float,
    time_to_expiry: float,
    N_paths: int = 100_000,
    J: int = 10,
) -> Dict[str, float]:
    """
    Estimate the true probability of a Polymarket binary contract
    using the full stacked variance reduction pipeline.

    current_price: Current YES price (0-1)
    vol_estimate: Estimated annualized volatility of the logit price
    time_to_expiry: Time to contract expiry in years

    Returns probability estimate with confidence interval.
    """
    # For a binary contract, we simulate in probability space
    # S0 = current_price, K = 0.5 (resolved YES if final > 0.5)
    # But we need to account for the logit dynamics

    from scipy.special import logit as logit_fn, expit

    # Work in logit space for bounded simulation
    x0 = logit_fn(np.clip(current_price, 0.001, 0.999))

    n_per_stratum = N_paths // J
    n_half = n_per_stratum // 2

    all_terminal_probs = []

    for j in range(J):
        U = np.random.uniform(j / J, (j + 1) / J, n_half)
        Z = norm.ppf(U)
        Z_combined = np.concatenate([Z, -Z])  # Antithetic

        # Logit random walk terminal values
        x_T = x0 + (-0.5 * vol_estimate**2) * time_to_expiry + vol_estimate * np.sqrt(time_to_expiry) * Z_combined

        # Transform back to probability space
        p_T = expit(x_T)
        all_terminal_probs.extend(p_T)

    all_terminal_probs = np.array(all_terminal_probs)

    # Probability that contract resolves YES (terminal prob > 0.5)
    resolves_yes = (all_terminal_probs > 0.5).astype(float)

    p_estimate = float(resolves_yes.mean())
    se = float(resolves_yes.std() / np.sqrt(len(resolves_yes)))

    return {
        "probability_estimate": p_estimate,
        "std_error": se,
        "ci_95": (p_estimate - 1.96 * se, p_estimate + 1.96 * se),
        "current_price": current_price,
        "edge": p_estimate - current_price,
        "edge_in_se": (p_estimate - current_price) / se if se > 0 else 0.0,
        "vol_estimate": vol_estimate,
        "time_to_expiry": time_to_expiry,
    }
