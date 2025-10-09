from classic_optimizer import MeanVarianceMarkowitz
from utils import mean_cov,annualize



if __name__ == "__main__":
    
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    np.random.seed(7)
    n_assets = 6
    T = 600  # observations

#     # # --- Replace this block with your CSV prices ---
#     # # Example synthetic returns with a random covariance
    A = np.random.randn(n_assets, n_assets)
    print(A)
    print(A.shape)
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

    mvm = MeanVarianceMarkowitz(mu=mu, Sigma=Sigma, rf=rf, short_sale=True)
    print(mvm.gmv)
   