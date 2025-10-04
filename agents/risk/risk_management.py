import pandas as pd
import numpy as np
from typing import List, Dict
from risk_measurement import (
    standard_deviation, 
    value_at_risk, 
    max_drawdown, 
    sharpe_ratio,
    correlation_matrix
    )

"""
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


def calculate_risk_metrics(returns) -> dict:
    risk_metrics = {
        "Standard_Deviation": standard_deviation(returns),
        "Value_at_Risk": value_at_risk(returns),
        "Max_Drawdown": max_drawdown(returns),
        "Sharpe_Ratio": sharpe_ratio(returns),
        # "Correlation_Matrix": correlation_matrix(returns)
    }
    return risk_metrics

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


if __name__ == "__main__":  
    from pathlib import Path
    import pandas as pd

    file_path = Path.cwd()/"data"/"archive"/"stocks"/"AAPL.csv"
    df = pd.read_csv(file_path)
    print(df.head())
    
    df = df.sort_values(by="Date")
    
    df["Return"] = df["Close"].pct_change()
    
    df = df.dropna()
    
    returns = df["Return"]

    print("Calculating Risk Metrics...")
    risk_metrics = calculate_risk_metrics(returns)
    for metric, value in risk_metrics.items():
        print(f"{metric}: {value}")

    print("Identifying Risks...",)