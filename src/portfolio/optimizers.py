import numpy as np

class ClassicOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def mean_variance_step(self, weights, returns, cov_matrix):
        # Gradient of the mean-variance objective
        grad = -returns + cov_matrix @ weights
        return weights - self.learning_rate * grad
    
    def minimum_variance_step(self, weights, cov_matrix):
        # Gradient of the minimum-variance objective
        grad = cov_matrix @ weights
        return weights - self.learning_rate * grad
    
    def risk_parity_step(self, weights, cov_matrix):
        # Gradient of the risk-parity objective
        portfolio_variance = weights.T @ cov_matrix @ weights
        marginal_risk_contribution = cov_matrix @ weights
        risk_contribution = weights * marginal_risk_contribution
        target = portfolio_variance / len(weights)
        grad = 2 * (risk_contribution - target) / (weights + 1e-8)
        return weights - self.learning_rate * grad
    
    def black_litterman_step(self, weights, pi, tau, P, Q, cov_matrix):
        # Gradient of the Black-Litterman objective
        inv_cov = np.linalg.inv(cov_matrix)
        M = np.linalg.inv(inv_cov + (P.T @ np.linalg.inv(tau * np.eye(len(Q))) @ P))
        adjusted_returns = M @ (inv_cov @ pi + P.T @ np.linalg.inv(tau * np.eye(len(Q))) @ Q)
        grad = -adjusted_returns + cov_matrix @ weights
        return weights - self.learning_rate * grad  

class AlgorithmicOptimizer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, gradients):
        raise NotImplementedError("This method should be overridden by subclasses.")