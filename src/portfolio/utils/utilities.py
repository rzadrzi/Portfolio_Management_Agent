# mean_variance_markowitz.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def to_returns(price_df: pd.DataFrame, log=True):
    if log:
        rets = np.log(price_df / price_df.shift(1)).dropna()
    else:
        rets = price_df.pct_change().dropna()
    return rets

def mean_cov(returns: pd.DataFrame):
    mu = returns.mean().values  # arithmetic mean per period
    Sigma = returns.cov().values
    return mu, Sigma

def annualize(mu, Sigma, periods_per_year=252):
    mu_a = mu * periods_per_year
    Sigma_a = Sigma * periods_per_year
    return mu_a, Sigma_a

def portfolio_stats(w, mu, Sigma, rf=0.0):
    er = float(w @ mu)
    var = float(w @ Sigma @ w)
    vol = np.sqrt(var)
    sharpe = (er - rf) / vol if vol > 0 else np.nan
    return er, vol, sharpe

def project_to_simplex(w):
    # ensure sum to 1 and nonnegative (for numerical clean-up)
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else w

def minimize_vol(target_return, mu, Sigma):
    n = len(mu)
    w0 = np.ones(n) / n  # initial guess: equal weights

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
        {'type': 'eq', 'fun': lambda w: w @ mu - target_return}  # target return
    )
    bounds = tuple((0, 1) for _ in range(n))  # no short selling

    result = minimize(lambda w: w @ Sigma @ w, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else None