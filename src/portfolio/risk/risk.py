import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from typing import List, Dict, Any
import yfinance as yf


"""
Risk analysis module for financial portfolios.

1.Common Quantitative Methods for Risk Analysis:

    Standard Deviation: 
        Measures the dispersion of an investment's returns from its average, indicating its volatility and associated risk. 
    Value-at-Risk (VaR): 
        Estimates the maximum potential loss over a specific time period at a given confidence level, providing a worst-case scenario outlook. 
    Tracking Error: 
        A measure of active risk, it shows how much a portfolio's returns deviate from its benchmark index. 
    Drawdown: 
        Measures the peak-to-trough decline in a portfolio's value over a specific period, highlighting the magnitude of losses from its highest point. 
    Beta: 
        Quantifies the sensitivity of a portfolio's returns to overall market movements. 
    Correlation Analysis: 
        Examines the relationships between different assets within a portfolio to understand how they move together, which affects diversification and risk. 
  
2.Advanced Analytical Techniques:

    Stress Testing: 
        Involves simulating extreme market events or adverse conditions to gauge the impact on the portfolio's value and identify vulnerabilities. 
    Scenario Analysis: 
        "What-if" analyses that assess the potential impact of various prospective trades, hedges, or market conditions on risk and return. 
    Factor Decomposition: 
        Breaking down portfolio risk into its constituent drivers (e.g., macroeconomic factors) to understand the underlying sources of risk and their relative contributions. 
    Marginal VaR and Component VaR: 
        These tools help to identify which specific positions contribute most to the overall portfolio risk, enabling targeted risk reduction strategies. 

3.Qualitative and Strategic Approaches:

    Risk Identification: 
        The first step in risk management, which involves recognizing potential risks associated with investments and market conditions. 
    Weighted Ranking and Scoring Techniques: 
        Used by governing bodies to assess risks by assigning weights and scores, helping to prioritize risks during governance meetings. 
    Portfolio Diversification: 
        A strategic approach to spread investments across different asset classes, sectors, and geographies to reduce the impact of any single underperforming investment. 
    Asset Allocation: 
        A strategy that involves strategically distributing investments across different asset classes to manage risk according to the investor's goals. 
    Hedging: 
        Employing financial instruments or derivatives to offset potential losses in another investment, reducing overall portfolio risk.       

A comprehensive risk analysis and management tool for financial portfolios.
This class provides methods for various risk analysis techniques, including both common quantitative methods and advanced analytical techniques.
It also includes qualitative and strategic approaches to risk management.
Note:
    This class is designed for educational and illustrative purposes. In a production environment,
    additional error handling, logging, and validation may be necessary.
"""


"""
==================================================
            Risk Measurement
==================================================
"""
"""
Functions:
    standard_deviation(returns): Calculate the standard deviation of returns.
    value_at_risk(returns, confidence_level): Calculate the Value-at-Risk (VaR) at a given confidence level.
    tracking_error(portfolio_returns, benchmark_returns): Calculate the tracking error between portfolio and benchmark returns.
    drawdown(portfolio_values): Calculate the drawdown of a portfolio.
    beta(portfolio_returns, market_returns): Calculate the beta of a portfolio relative to the market.
    correlation_matrix(returns): Calculate the correlation matrix of asset returns.
    factor_decomposition(returns, factors): Decompose portfolio returns into factor contributions.
    marginal_var(portfolio_returns, asset_returns, confidence_level): Calculate the Marginal VaR for each asset in the portfolio.
    component_var(portfolio_returns, asset_returns, confidence_level): Calculate the Component VaR for each asset in the portfolio.
    plot_drawdown(drawdown): Plot the drawdown of a portfolio.
    plot_correlation_matrix(returns): Plot the correlation matrix of asset returns.
    asset_allocation(total_investment, allocation): Allocate total investment across different asset classes.
    risk_identification(portfolio): Identify potential risks in the portfolio.
    weighted_ranking(risks, weights): Rank risks based on weighted scoring.
    governance_meeting(risks, weights): Conduct a governance meeting to discuss and prioritize risks.
    max_drawdown(portfolio_values): Calculate the maximum drawdown of a portfolio.
    sharpe_ratio(returns, risk_free_rate): Calculate the Sharpe Ratio of a portfolio.
    calmar_ratio(portfolio_values): Calculate the Calmar Ratio of a portfolio.
    sesortino_ratio(returns, risk_free_rate): Calculate the Sortino Ratio of a portfolio.
    treynor_ratio(returns, market_returns, risk_free_rate): Calculate the Treyn
    

"""
def standard_deviation( returns: pd.Series) -> float:
    """Calculate the standard deviation of returns."""
    return np.std(returns)

def value_at_risk( returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate the Value-at-Risk (VaR) at a given confidence level."""
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    mean = np.mean(returns)
    std_dev = np.std(returns)
    var = norm.ppf(1 - confidence_level, mean, std_dev)
    return var

def tracking_error( portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate the tracking error between portfolio and benchmark returns."""
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio and benchmark returns must have the same length.")
    return np.std(portfolio_returns - benchmark_returns)

def drawdown( portfolio_values: pd.Series) -> pd.Series:
    """Calculate the drawdown of a portfolio."""
    peak = portfolio_values.cummax()
    dd = (portfolio_values - peak) / peak
    return dd

def beta( portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate the beta of a portfolio relative to the market."""
    if len(portfolio_returns) != len(market_returns):
        raise ValueError("Portfolio and market returns must have the same length.")
    covariance = np.cov(portfolio_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    return beta

def correlation_matrix( returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate the correlation matrix of asset returns."""
    return returns.corr()

def factor_decomposition( returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Decompose portfolio returns into factor contributions."""
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    model.fit(factors, returns)
    factor_contributions = pd.DataFrame(model.coef_, index=factors.columns, columns=['Contribution'])
    return factor_contributions

def marginal_var( portfolio_returns: pd.Series, asset_returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.Series:
    """Calculate the Marginal VaR for each asset in the portfolio."""
    base_var = value_at_risk(portfolio_returns, confidence_level)
    marginal_vars = {}
    for asset in asset_returns.columns:
        perturbed_portfolio = portfolio_returns + asset_returns[asset]
        perturbed_var = value_at_risk(perturbed_portfolio, confidence_level)
        marginal_vars[asset] = perturbed_var - base_var
    return pd.Series(marginal_vars)

def component_var( portfolio_returns: pd.Series, asset_returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.Series:
    """Calculate the Component VaR for each asset in the portfolio."""
    total_var = value_at_risk(portfolio_returns, confidence_level)
    marginal_vars = marginal_var(portfolio_returns, asset_returns, confidence_level)
    weights = asset_returns.mean() / asset_returns.mean().sum()
    component_vars = weights * marginal_vars
    return component_vars

def plot_drawdown( drawdown: pd.Series):
    """Plot the drawdown of a portfolio."""
    plt.figure(figsize=(10, 6))
    plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.5)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid()
    plt.show()

def plot_correlation_matrix( returns: pd.DataFrame):
    """Plot the correlation matrix of asset returns."""
    corr_matrix = correlation_matrix(returns)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Asset Returns')
    plt.show()

def weighted_ranking( risks: List[str], weights: Dict[str, float]) -> pd.Series:
    """Rank risks based on weighted scoring."""
    scores = pd.Series({risk: weights.get(risk, 0) for risk in risks})
    return scores.sort_values(ascending=False)

def governance_meeting( risks: List[str], weights: Dict[str, float]) -> pd.Series:
    """Conduct a governance meeting to discuss and prioritize risks."""
    ranked_risks = weighted_ranking(risks, weights)
    return ranked_risks

def max_drawdown( portfolio_values: pd.Series) -> float:
    """Calculate the maximum drawdown of a portfolio."""
    dd = drawdown(portfolio_values)
    max_dd = dd.min()
    return max_dd

def sharpe_ratio( returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe Ratio of a portfolio."""
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else np.nan

def sortino_ratio( returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sortino Ratio of a portfolio."""
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns)
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / downside_std if downside_std != 0 else np.nan

def calmar_ratio( portfolio_values: pd.Series) -> float:
    """Calculate the Calmar Ratio of a portfolio."""
    annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252 / len(portfolio_values)) - 1
    max_dd = abs(max_drawdown(portfolio_values))
    # print(f"Annual Return: {annual_return}, Max Drawdown: {max_dd}")
    return annual_return / max_dd if max_dd != 0 else np.nan

def treynor_ratio( returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Treynor Ratio of a portfolio."""
    bb = beta(returns, market_returns)
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / bb if bb != 0 else np.nan  

def expected_shortfall( returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate the Expected Shortfall (Conditional VaR) at a given confidence level."""
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    expected_shortfall = returns[returns <= var_threshold].mean()
    return expected_shortfall


"""
==================================================
            Risk Management
==================================================
"""

"""

Functions:

    rebalance_portfolio(current_weights, target_weights, threshold): Rebalance the portfolio if weights deviate from target weights by a certain threshold.
    diversify_portfolio(assets, max_weight_per_asset): Create a diversified portfolio with maximum weight per asset.
    hedge_portfolio(portfolio_values, hedge_ratio): Hedge the portfolio by a certain ratio.
    scenario_analysis(portfolio_values, scenarios): Perform scenario analysis by applying different scenarios to the portfolio values.
    stress_test(portfolio_values, shock): Perform a stress test by applying a shock to the portfolio values.
    risk_identification(portfolio): Identify potential risks in the portfolio.
    weighted_ranking(risks, weights): Rank risks based on weighted scoring.
    governance_meeting(risks, weights): Conduct a governance meeting to discuss and prioritize
    asset_allocation(total_investment, allocation): Allocate total investment across different asset classes.  

"""
"""
Chosen risk analysis and management techniques based on portfolio characteristics and investment goals.

1.standard_deviation(returns["AAPL"])
2.value_at_risk(returns["AAPL"])
3.max_drawdown(portfolio_values)
4.sharpe_ratio(returns["AAPL"])
5.correlation_matrix(returns)

"""

def risk_identification( portfolio: pd.DataFrame) -> List[str]:
    """Identify potential risks in the portfolio."""
    risks = []
    if portfolio.isnull().values.any():
        risks.append("Data Quality Risk: Missing values in portfolio data.")
    if (portfolio < 0).any().any():
        risks.append("Negative Holdings Risk: Portfolio contains negative holdings.")
    if portfolio.shape[1] < 2:
        risks.append("Concentration Risk: Portfolio has less than two assets.")
    return risks    

def weighted_ranking( risks: List[str], weights: Dict[str, float]) -> pd.Series:
    """Rank risks based on weighted scoring."""
    scores = pd.Series({risk: weights.get(risk, 0) for risk in risks})
    return scores.sort_values(ascending=False)

def diversify_portfolio( assets: List[str], max_weight_per_asset: float = 0.35) -> pd.Series:
    """Create a diversified portfolio with maximum weight per asset."""
    num_assets = len(assets)
    if num_assets == 0:
        raise ValueError("Asset list cannot be empty.")
    equal_weight = min(1.0 / num_assets, max_weight_per_asset)
    weights = pd.Series(equal_weight, index=assets)
    weights /= weights.sum()  # Normalize to sum to 1
    return weights

def asset_allocation( total_investment: float, allocation: Dict[str, float]) -> pd.Series:
    """Allocate total investment across different asset classes."""
    if not np.isclose(sum(allocation.values()), 1.0):
        raise ValueError("Allocation percentages must sum to 1.")
    allocation_series = pd.Series({asset: total_investment * weight for asset, weight in allocation.items()})
    return allocation_series

def hedge_portfolio( portfolio_values: pd.Series, hedge_ratio: float) -> pd.Series:
    """Hedge the portfolio by a certain ratio."""
    if not 0 <= hedge_ratio <= 1:
        raise ValueError("Hedge ratio must be between 0 and 1.")
    hedged_values = portfolio_values * (1 - hedge_ratio)
    return hedged_values

def rebalance_portfolio( current_weights: pd.Series, target_weights: pd.Series, threshold: float = 0.05) -> pd.Series:
    """Rebalance the portfolio if weights deviate from target weights by a certain threshold."""
    deviation = (current_weights - target_weights).abs()
    if (deviation > threshold).any():
        return target_weights
    return current_weights

def scenario_analysis( portfolio_values: pd.Series, scenarios: Dict[str, float]) -> Dict[str, pd.Series]:
    """Perform scenario analysis by applying different scenarios to the portfolio values."""
    results = {}
    for scenario, impact in scenarios.items():
        results[scenario] = portfolio_values * (1 + impact)
    return results

def stress_test( portfolio_values: pd.Series, shock: float) -> pd.Series:
    """Perform a stress test by applying a shock to the portfolio values."""
    stressed_values = portfolio_values * (1 - shock)
    return stressed_values 

def governance_meeting( risks: List[str], weights: Dict[str, float]) -> pd.Series:
    """Conduct a governance meeting to discuss and prioritize risks."""
    ranked_risks = weighted_ranking(risks, weights)
    print("Governance Meeting - Ranked Risks:")
    for risk, score in ranked_risks.items():
        print(f"Risk: {risk}, Score: {score}")
    return ranked_risks




















# Example usage:
if __name__ == "__main__":  
    from pathlib import Path
    import pandas as pd

    file_path = Path.cwd()/"data"/"archive"/"stocks"/"AAPL.csv"
    df = pd.read_csv(file_path)
    print(df.head())
    
    df = df.sort_values(by="Date")
    
    df["Return"] = df["Close"].pct_change()
    
    df = df.dropna()
    
    print("Standard Deviation:", standard_deviation(df["Return"]))
    print("Value at Risk (95%):", value_at_risk(df["Return"], 0.95))
    print("Drawdown:", drawdown(df["Close"]).tail())
    print("Beta (vs S&P 500):", beta(df["Return"], df["Return"]))  # Replace with actual market returns
    print("Sharpe Ratio:", sharpe_ratio(df["Return"], risk_free_rate=0.01))
    print("Sortino Ratio:", sortino_ratio(df["Return"], risk_free_rate=0.01))
    print("Correlation Matrix:\n", correlation_matrix(df[["Return"]]))
    print("Max Drawdown:", max_drawdown(df["Close"]))
    # calmar_ratio(df["Close"])
    print("Calmar Ratio:", calmar_ratio(df["Close"]))
    print("Treynor Ratio:", treynor_ratio(df["Return"], df["Return"], risk_free_rate=0.01))  # Replace with actual market returns

    # Example of plotting drawdown
    # dd = drawdown(df["Close"])
    # plot_drawdown(dd)
    # Example of plotting correlation matrix
    # plot_correlation_matrix(df[["Return"]])      