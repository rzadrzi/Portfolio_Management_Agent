# utilities.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .utilities import to_returns, mean_cov, annualize, portfolio_stats, project_to_simplex

# ---------- Closed-form (no short-sale) ----------
def gmv_weights_closed(Sigma):
    invS = np.linalg.inv(Sigma)
    ones = np.ones(Sigma.shape[0])
    w = invS @ ones
    w /= ones @ invS @ ones
    return w

def tangency_weights_closed(mu, Sigma, rf=0.0):
    invS = np.linalg.inv(Sigma)
    ones = np.ones(Sigma.shape[0])
    excess = mu - rf * ones
    w = invS @ excess
    w /= ones @ invS @ excess
    return w

# ---------- Numerical (with short-sale) ----------
# ---------- Numerical QP (handles no-short & extra constraints) ----------
def min_variance_given_return(mu, Sigma, target_return, short_sale=True):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = None if short_sale else [(0.0, 1.0)] * n
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w, mu=mu: w @ mu - target_return}
    )
    def obj(w): return w @ Sigma @ w
    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x

def global_min_variance(Sigma, short_sale=True):
    n = Sigma.shape[0]
    w0 = np.ones(n)/n
    bounds = None if short_sale else [(0.0, 1.0)]*n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    res = minimize(lambda w: w @ Sigma @ w, w0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"GMV failed: {res.message}")
    return res.x

def tangency_portfolio(mu, Sigma, rf=0.0, short_sale=True):
    n = len(mu)
    w0 = np.ones(n)/n
    bounds = None if short_sale else [(0.0, 1.0)] * n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    def neg_sharpe(w):
        er, vol, sr = portfolio_stats(w, mu, Sigma, rf)
        return -sr
    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Tangency failed: {res.message}")
    return res.x

def efficient_frontier(mu, Sigma, n_points=50, short_sale=True):
    # pick a reasonable return range
    min_w = global_min_variance(Sigma, short_sale=short_sale)
    r_min, _, _ = portfolio_stats(min_w, mu, Sigma)
    # a "max return" portfolio by putting all weight to best mu (if short not allowed)
    if short_sale:
        r_max = float(np.max(mu))
    else:
        idx = np.argmax(mu)
        unit = np.zeros_like(mu); unit[idx] = 1.0
        r_max = float(unit @ mu)
    targets = np.linspace(r_min, r_max, n_points)
    ws, ers, vols = [], [], []
    for t in targets:
        try:
            w = min_variance_given_return(mu, Sigma, t, short_sale=short_sale)
            er, vol, _ = portfolio_stats(w, mu, Sigma)
            ws.append(w); ers.append(er); vols.append(vol)
        except RuntimeError:
            # infeasible target under no-short; skip
            continue
    return np.array(ws), np.array(ers), np.array(vols)
