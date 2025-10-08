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


# ---------- Demo / Example ----------
if __name__ == "__main__":
    np.random.seed(7)
    n_assets = 6
    T = 600  # observations

    # --- Replace this block with your CSV prices ---
    # Example synthetic returns with a random covariance
    A = np.random.randn(n_assets, n_assets)
    Sigma_true = A @ A.T / n_assets
    mu_true = np.array([0.10, 0.12, 0.08, 0.15, 0.05, 0.11]) / 252  # daily mean
    R = np.random.multivariate_normal(mean=mu_true, cov=Sigma_true/252, size=T)
    returns = pd.DataFrame(R, columns=[f"A{i+1}" for i in range(n_assets)])
    # --- If you have prices, do this instead:
    # prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    # returns = to_returns(prices, log=True)

    mu, Sigma = mean_cov(returns)
    mu_a, Sigma_a = annualize(mu, Sigma, periods_per_year=252)

    rf = 0.02  # 2% annual risk-free

    # Closed-form (allow short-selling)
    w_gmv = gmv_weights_closed(Sigma_a)
    w_tan = tangency_weights_closed(mu_a, Sigma_a, rf=rf)

    # Numerical (no short-selling)
    w_gmv_ns = global_min_variance(Sigma_a, short_sale=False)
    w_tan_ns = tangency_portfolio(mu_a, Sigma_a, rf=rf, short_sale=False)

    # Efficient frontier
    W, ERs, VOLs = efficient_frontier(mu_a, Sigma_a, n_points=60, short_sale=False)

    # Print key portfolios
    print("GMV (closed-form, short allowed):", np.round(w_gmv, 4))
    print("Tangency (closed-form, short allowed):", np.round(w_tan, 4))
    er_gmv, vol_gmv, sr_gmv = portfolio_stats(w_gmv, mu_a, Sigma_a, rf)
    er_tan, vol_tan, sr_tan = portfolio_stats(w_tan, mu_a, Sigma_a, rf)
    print(f"GMV  -> E[R]={er_gmv:.3%}, Vol={vol_gmv:.3%}, Sharpe={sr_gmv:.3f}")
    print(f"TAN  -> E[R]={er_tan:.3%}, Vol={vol_tan:.3%}, Sharpe={sr_tan:.3f}")

    # Plot Efficient Frontier
    plt.figure(figsize=(7,5))
    plt.scatter(VOLs, ERs, s=12, label="Efficient Frontier (No-Short)")
    # mark GMV & Tangency (no-short)
    er_gmv_ns, vol_gmv_ns, _ = portfolio_stats(w_gmv_ns, mu_a, Sigma_a, rf)
    er_tan_ns, vol_tan_ns, _ = portfolio_stats(w_tan_ns, mu_a, Sigma_a, rf)
    plt.scatter([vol_gmv_ns],[er_gmv_ns], marker='o', s=60, label="GMV (No-Short)")
    plt.scatter([vol_tan_ns],[er_tan_ns], marker='^', s=60, label="Tangency (No-Short)")

    plt.xlabel("Volatility (annualized)")
    plt.ylabel("Expected Return (annualized)")
    plt.legend()
    plt.title("Markowitz Mean-Variance: Efficient Frontier")
    plt.tight_layout()
    plt.show()

    # Show weights table for tangency (no-short)
    weights_df = pd.DataFrame({
        "Asset": returns.columns,
        "w_Tangency_NoShort": np.round(w_tan_ns, 4),
        "w_GMV_NoShort": np.round(w_gmv_ns, 4)
    })
    print(weights_df)
