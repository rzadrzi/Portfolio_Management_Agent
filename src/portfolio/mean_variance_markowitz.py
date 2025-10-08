# utilities.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .utilities import to_returns, mean_cov, annualize, portfolio_stats, project_to_simplex


class MeanVarianceMarkowitz:
    def __init__(self, mu, Sigma, rf=0.1, short_sale=True):
        self.mu = mu
        self.Sigma = Sigma
        self.rf = rf
        self.short_sale = short_sale

    def gmv(self):
        """Global Minimum Variance Portfolio"""
        if not self.short_sale:
            # Closed-form (no short-sale)
            invS = np.linalg.inv(self.Sigma)
            ones = np.ones(self.Sigma.shape[0])
            w = invS @ ones
            w /= ones @ invS @ ones
            return w
        else:
            n = self.Sigma.shape[0]
            w0 = np.ones(n)/n
            bounds = None if self.short_sale else [(0.0, 1.0)]*n
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
            res = minimize(lambda w: w @ self.Sigma @ w, w0, method='SLSQP', bounds=bounds, constraints=cons)
            if not res.success:
                raise RuntimeError(f"GMV failed: {res.message}")
            return res.x
        
    def tangency(self):
        """Tangency Portfolio / Maximum Sharpe Ratio"""
        if not self.short_sale:
            # Closed-form (no short-sale)
            invS = np.linalg.inv(self.Sigma)
            ones = np.ones(self.Sigma.shape[0])
            excess = self.mu - self.rf * ones
            w = invS @ excess
            w /= ones @ invS @ excess
            return w
        else:
            n = len(self.mu)
            w0 = np.ones(n)/n
            bounds = None if self.short_sale else [(0.0, 1.0)]*n
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
            def neg_sharpe(w):
                er, vol, sr = portfolio_stats(w, self.mu, self.Sigma, self.rf)
                return -sr
            res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
            if not res.success:
                raise RuntimeError(f"Tangency failed: {res.message}")
            return res.x
        
    def min_variance(self):
        """Minimum Variance for Target Return"""
        n = len(self.mu)
        w0 = np.ones(n) / n
        bounds = None if self.short_sale else [(0.0, 1.0)] * n
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: w @ self.mu - self.rf}
        )
        def obj(w): return w @ self.Sigma @ w
        res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        return res.x
    
    def efficient_frontier(self, n_points=50):
        # pick a reasonable return range
        min_w = self.gmv()
        r_min, _, _ = portfolio_stats(min_w, self.mu, self.Sigma)
        # a "max return" portfolio by putting all weight to best mu (if short not allowed)
        if self.short_sale:
            r_max = float(np.max(self.mu))
        else:
            idx = np.argmax(self.mu)
            unit = np.zeros_like(self.mu); unit[idx] = 1.0
            r_max = float(unit @ self.mu)
        targets = np.linspace(r_min, r_max, n_points)
        ws, ers, vols = [], [], []
        for t in targets:
            try:
                w = self.min_variance()
                er, vol, _ = portfolio_stats(w, self.mu, self.Sigma)
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


    mvm = MeanVarianceMarkowitz(mu_a, Sigma_a, rf=rf, short_sale=False)
    w_gmv = mvm.gmv()
    w_tan = mvm.tangency()

    print("GMV Weights (no short-sale):", np.round(w_gmv, 4))
    print("Tangency Weights (no short-sale):", np.round(w_tan, 4))
    
    er_gmv, vol_gmv, sr_gmv = portfolio_stats(w_gmv, mu_a, Sigma_a, rf)
    er_tan, vol_tan, sr_tan = portfolio_stats(w_tan, mu_a, Sigma_a, rf)
    
    print(f"GMV  -> E[R]={er_gmv:.3%}, Vol={vol_gmv:.3%}, Sharpe={sr_gmv:.3f}")
    print(f"TAN  -> E[R]={er_tan:.3%}, Vol={vol_tan:.3%}, Sharpe={sr_tan:.3f}")
    
    W, ERs, VOLs = mvm.efficient_frontier(n_points=60)
    # Plot Efficient Frontier
    plt.figure(figsize=(7,5))
    plt.scatter(VOLs, ERs, s=12, label="Efficient Frontier (No-Short)")
    # mark GMV & Tangency (no-short)
    plt.scatter([vol_gmv],[er_gmv], marker='o', s=60, label="GMV (No-Short)")
    plt.scatter([vol_tan],[er_tan], marker='^', s=60, label="Tangency (No-Short)")  
    plt.xlabel("Volatility (annualized)")
    plt.ylabel("Expected Return (annualized)")
    plt.legend()
    plt.title("Markowitz Mean-Variance: Efficient Frontier")
    plt.tight_layout()
    plt.show()
# mean_variance_markowitz.py
# import numpy as np    