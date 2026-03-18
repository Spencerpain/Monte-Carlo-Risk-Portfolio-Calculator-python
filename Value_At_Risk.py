import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MonteCarlo:
    @staticmethod
    def run_simulation(weights, mean_returns, cov_matrix, portfolio_value, days, simulations):
        """Monte Carlo simulation — returns (days x simulations) matrix of portfolio values."""
        meanM = np.full(shape=(days, len(weights)), fill_value=mean_returns).T
        portfolio_sims = np.full(shape=(days, simulations), fill_value=0.0)

        for m in range(simulations):
            Z = np.random.normal(size=(days, len(weights)))
            L = np.linalg.cholesky(cov_matrix)
            daily_returns = meanM + np.inner(L, Z)
            portfolio_sims[:, m] = (
                np.cumprod(np.inner(weights, daily_returns.T) + 1) * portfolio_value
            )

        return portfolio_sims

    @staticmethod
    def calculate_var(portfolio_sims, confidence_interval, initial_portfolio):
        """VaR from simulated final portfolio values."""
        port_results = portfolio_sims[-1, :]
        VaR = initial_portfolio - np.percentile(port_results, (1 - confidence_interval) * 100)
        return VaR

    @staticmethod
    def calculate_es(portfolio_sims, confidence_interval, initial_portfolio):
        """Expected Shortfall (CVaR): mean loss beyond the VaR threshold."""
        port_results = portfolio_sims[-1, :]
        var_threshold = np.percentile(port_results, (1 - confidence_interval) * 100)
        tail = port_results[port_results <= var_threshold]
        if len(tail) == 0:
            return 0.0
        return initial_portfolio - np.mean(tail)

    @staticmethod
    def calculate_component_es(weights, returns, confidence_interval, portfolio_value):
        """
        Component ES per asset: each asset's average contribution to losses
        in the tail scenarios. Used for limit attribution on risk desks.
        """
        weights = np.array(weights)
        port_returns = returns.values @ weights
        threshold = np.percentile(port_returns, (1 - confidence_interval) * 100)
        tail_mask = port_returns <= threshold
        if tail_mask.sum() == 0:
            return pd.Series(0.0, index=returns.columns)
        tail_returns = returns.values[tail_mask]
        component_es = pd.Series(
            -weights * tail_returns.mean(axis=0) * portfolio_value,
            index=returns.columns,
        )
        return component_es

    @staticmethod
    def calculate_risk_decomposition(weights, returns, confidence_interval, portfolio_value):
        """
        Marginal VaR and Contribution VaR per asset.
        Marginal VaR_i = corr(asset_i, portfolio) * vol_i / port_vol * portfolio_VaR_pct
        Component VaR_i = w_i * Marginal VaR_i * portfolio_value
        """
        weights = np.array(weights)
        port_returns = returns.values @ weights
        var_pct = abs(np.percentile(port_returns, (1 - confidence_interval) * 100))

        port_vol = np.std(port_returns)
        asset_vols = returns.std().values

        corr_with_port = np.array(
            [np.corrcoef(returns.values[:, i], port_returns)[0, 1] for i in range(len(weights))]
        )

        marginal_var = corr_with_port * asset_vols / (port_vol + 1e-12) * var_pct
        component_var = weights * marginal_var * portfolio_value
        total_component_var = component_var.sum()

        result = pd.DataFrame(
            {
                "Weight (%)": weights * 100,
                "Marginal VaR (%)": marginal_var * 100,
                "Component VaR ($)": component_var,
                "Component VaR (%)": component_var / (total_component_var + 1e-12) * 100,
            },
            index=returns.columns,
        )
        return result


class PortfolioOptimizer:
    @staticmethod
    def max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.05 / 252):
        """Maximize Sharpe ratio (long-only)."""
        n = len(mean_returns)

        def neg_sharpe(w):
            ret = np.dot(w, mean_returns)
            vol = np.sqrt(w @ cov_matrix.values @ w)
            return -(ret - risk_free_rate) / (vol + 1e-12)

        result = minimize(
            neg_sharpe,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        return result.x

    @staticmethod
    def min_variance(mean_returns, cov_matrix):
        """Minimum variance portfolio (long-only)."""
        n = len(mean_returns)

        result = minimize(
            lambda w: np.sqrt(w @ cov_matrix.values @ w),
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        return result.x

    @staticmethod
    def min_cvar(returns, confidence_interval):
        """Minimize CVaR directly using simulated historical scenarios."""
        n = returns.shape[1]
        scenarios = returns.values

        def portfolio_cvar(w):
            port_returns = scenarios @ w
            threshold = np.percentile(port_returns, (1 - confidence_interval) * 100)
            tail = port_returns[port_returns <= threshold]
            return -np.mean(tail) if len(tail) > 0 else 0.0

        result = minimize(
            portfolio_cvar,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        return result.x

    @staticmethod
    def risk_parity(cov_matrix):
        """Risk parity: each asset contributes equally to total portfolio volatility."""
        n = cov_matrix.shape[0]
        cov = cov_matrix.values

        def risk_parity_obj(w):
            port_vol = np.sqrt(w @ cov @ w)
            marginal = (cov @ w) / (port_vol + 1e-12)
            risk_contrib = w * marginal
            target = port_vol / n
            return np.sum((risk_contrib - target) ** 2)

        result = minimize(
            risk_parity_obj,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0.01, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        return result.x

    @staticmethod
    def efficient_frontier(mean_returns, cov_matrix, n_points=60):
        """Compute the efficient frontier (return, vol) pairs."""
        n = len(mean_returns)
        cov = cov_matrix.values
        target_returns = np.linspace(float(mean_returns.min()), float(mean_returns.max()), n_points)
        frontier_vols = []

        for target in target_returns:
            result = minimize(
                lambda w: np.sqrt(w @ cov @ w),
                np.ones(n) / n,
                method="SLSQP",
                bounds=[(0, 1)] * n,
                constraints=[
                    {"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
                ],
            )
            frontier_vols.append(np.sqrt(result.x @ cov @ result.x) if result.success else np.nan)

        return target_returns, np.array(frontier_vols)

    @staticmethod
    def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate=0.05 / 252):
        """Return annualised return, vol, and Sharpe for a given weight vector."""
        weights = np.array(weights)
        ret = np.dot(weights, mean_returns) * 252
        vol = np.sqrt(weights @ cov_matrix.values @ weights) * np.sqrt(252)
        sharpe = (ret - risk_free_rate * 252) / (vol + 1e-12)
        return ret, vol, sharpe


class Backtester:
    @staticmethod
    def rolling_var_backtest(returns, weights, confidence_interval, window=252):
        """
        Roll a window through history, compute 1-day VaR each day,
        then compare to the actual next-day loss.
        Returns a DataFrame with VaR estimates, actual losses, and exceedance flags.
        """
        weights = np.array(weights)
        port_returns = returns.values @ weights
        dates = returns.index

        var_estimates, actual_losses, exceedances = [], [], []

        for i in range(window, len(port_returns)):
            window_rets = port_returns[i - window : i]
            var = -np.percentile(window_rets, (1 - confidence_interval) * 100)
            actual_loss = -port_returns[i]

            var_estimates.append(var)
            actual_losses.append(actual_loss)
            exceedances.append(bool(actual_loss > var))

        return pd.DataFrame(
            {"VaR": var_estimates, "Actual Loss": actual_losses, "Exceedance": exceedances},
            index=dates[window:],
        )

    @staticmethod
    def kupiec_test(exceedances, confidence_interval):
        """
        Kupiec traffic light test.
        Compares actual exceedance rate to expected (1 - CI).
        """
        n_obs = len(exceedances)
        n_exc = int(sum(exceedances))
        expected_rate = 1 - confidence_interval
        actual_rate = n_exc / n_obs if n_obs > 0 else 0.0

        if actual_rate <= expected_rate * 1.5:
            zone = "🟢 Green — Model OK"
        elif actual_rate <= expected_rate * 2.5:
            zone = "🟡 Yellow — Model Under Review"
        else:
            zone = "🔴 Red — Model Rejected"

        return {
            "Total Observations": n_obs,
            "Expected Exceedances": int(round(expected_rate * n_obs)),
            "Actual Exceedances": n_exc,
            "Expected Rate": f"{expected_rate:.1%}",
            "Actual Rate": f"{actual_rate:.1%}",
            "Traffic Light": zone,
        }
