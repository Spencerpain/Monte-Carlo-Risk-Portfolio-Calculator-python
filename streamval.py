import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import streamlit as st

from Get_Data import get_data  # expects: returns, mean_returns, cov_matrix


class MonteCarlo:
    @staticmethod
    def run_simulation(
        weights: np.ndarray,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        portfolio_value: float,
        days: int,
        simulations: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Monte Carlo simulation of portfolio value.
        Output shape: (days, simulations) with each column a simulated path.
        """
        w = np.asarray(weights, dtype=float)
        mu = np.asarray(mean_returns, dtype=float)              # (n,)
        Sigma = np.asarray(cov_matrix, dtype=float)             # (n,n)

        n = len(w)
        if mu.shape[0] != n or Sigma.shape != (n, n):
            raise ValueError("Dimension mismatch between weights, mean_returns, and cov_matrix.")

        # Cholesky factor once (Sigma must be positive definite)
        L = np.linalg.cholesky(Sigma)  # (n,n)

        rng = np.random.default_rng(seed)

        portfolio_sims = np.empty((days, simulations), dtype=float)

        for m in range(simulations):
            # Z: (days, n) iid N(0,1)
            Z = rng.standard_normal(size=(days, n))

            # Correlated shocks in return units: E = Z @ L.T  -> (days, n)
            E = Z @ L.T

            # Simulated asset returns per day: R = mu + E  -> broadcast to (days, n)
            R = E + mu

            # Portfolio daily return: rp = R @ w  -> (days,)
            rp = R @ w

            # Portfolio value path: V_t = V0 * cumprod(1 + rp_t)
            portfolio_sims[:, m] = portfolio_value * np.cumprod(1.0 + rp)

        return portfolio_sims

    @staticmethod
    def calculate_var(portfolio_sims: np.ndarray, confidence_interval: float, initial_portfolio: float) -> float:
        """
        VaR as a positive dollar loss number.
        VaR = V0 - percentile_{(1-c)}(V_T)
        """
        final_values = portfolio_sims[-1, :]  # all ending values
        cutoff = np.percentile(final_values, (1.0 - confidence_interval) * 100.0)
        return float(initial_portfolio - cutoff)


def main():
    st.set_page_config(
        page_title="Value at Risk Calculator",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Value At Risk Portfolio Calculator")
    st.write("##### Use this tool to understand the risk of your portfolio.")
    st.markdown("This tool is just a **SUGGESTION**, please invest at your own risk!")
    st.write("Click **Run Simulation** to begin.")

    st.sidebar.title("⚙️ Settings")

    with st.sidebar.expander("📊 Data Settings", expanded=True):
        # Robust ticker parsing (handles "SPY,QQQ" and "SPY, QQQ")
        tickers_raw = st.text_input(
            "Stock Tickers",
            "SPY, QQQ, SMH, GLD, TLT",
            help="Enter stock tickers separated by commas (e.g., SPY, QQQ, SMH, GLD, TLT)",
        )
        stock_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

        weights_input = st.text_input(
            "Portfolio Weights",
            "0.2, 0.2, 0.2, 0.2, 0.2",
            help="Enter portfolio weights separated by commas (must sum to 1).",
        )
        weights = np.array([float(w.strip()) for w in weights_input.split(",") if w.strip()], dtype=float)

        years_of_data = st.number_input(
            "Years of Historical Data",
            min_value=1,
            max_value=30,
            value=20,
            help="Select number of years to include in historical data (most recent n years)",
        )

        end_date = dt.datetime.now().strftime("%Y-%m-%d")
        start_date = (dt.datetime.now() - dt.timedelta(days=int(years_of_data) * 365)).strftime("%Y-%m-%d")

        portfolio_value = st.number_input("Initial Portfolio Value ($):", value=10000.0, help="Account balance")

    with st.sidebar.expander("🎲 Simulation Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            days = st.number_input(
                "Days to Simulate",
                min_value=2,
                max_value=1260,
                value=252,
                step=1,
                help="252 trading days ≈ 1 year",
            )

        with col2:
            simulations = st.number_input(
                "Simulations to Run",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Number of Monte Carlo trials",
            )

        confidence_interval = st.number_input(
            "Confidence Interval",
            value=0.95,
            min_value=0.90,
            max_value=0.99,
            help="Confidence level for VaR",
        )

        # Optional, helps reproducibility if you want it
        seed = st.number_input("Random Seed (optional)", value=0, step=1, help="Set >0 for repeatable results")
        seed_val = int(seed) if seed and int(seed) > 0 else None

    risk_level = 1.0 - confidence_interval

    if st.sidebar.button("Run Simulation"):
        try:
            # Validation
            if len(stock_list) == 0:
                st.error("Please enter at least one ticker.")
                st.stop()

            if len(weights) != len(stock_list):
                st.error("Number of weights must match number of tickers.")
                st.stop()

            if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                st.error(f"Weights must sum to 1. Current sum: {weights.sum():.6f}")
                st.stop()

            # Fetch real data
            returns, mean_returns, cov_matrix = get_data(stock_list, start=start_date, end=end_date)

            # Run simulation
            portfolio_sims = MonteCarlo.run_simulation(
                weights=weights,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                portfolio_value=float(portfolio_value),
                days=int(days),
                simulations=int(simulations),
                seed=seed_val,
            )

            # VaR
            VaR = MonteCarlo.calculate_var(portfolio_sims, confidence_interval, float(portfolio_value))

            settings_data = {
                "Parameter": [
                    "Stock Tickers",
                    "Portfolio Weights",
                    "Years of Data",
                    "Start Date",
                    "End Date (Date ran)",
                    "Initial Portfolio Value ($)",
                    "Days to Simulate",
                    "Number of Simulations",
                    "Confidence Interval",
                    "Risk Level",
                    "Value at Risk (VaR)",
                ],
                "Value": [
                    ", ".join(stock_list),
                    ", ".join(map(str, weights)),
                    int(years_of_data),
                    start_date,
                    end_date,
                    float(portfolio_value),
                    int(days),
                    int(simulations),
                    f"{confidence_interval * 100:.2f}%",
                    f"{risk_level * 100:.2f}%",
                    f"${VaR:.2f}",
                ],
            }
            settings_df = pd.DataFrame(settings_data).set_index("Parameter")

            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                st.write("### Simulation Results")
                fig, ax = plt.subplots(figsize=(6, 4))
                for i in range(int(simulations)):
                    ax.plot(portfolio_sims[:, i], alpha=0.25, linewidth=0.7)

                mean_simulation = np.mean(portfolio_sims, axis=1)
                ax.plot(mean_simulation, linewidth=2, label="Mean Simulation")
                ax.axhline(
                    y=(float(portfolio_value) - VaR),
                    linewidth=1,
                    label=f"VaR ({confidence_interval*100:.0f}%): ${VaR:.2f}",
                )

                ax.set_title(f"Simulated Portfolio Value over {int(days)} days ({int(simulations)} trials)")
                ax.set_xlabel("Days")
                ax.set_ylabel("Portfolio Value ($)")
                ax.legend(loc="upper left")
                st.pyplot(fig)

            with plot_col2:
                final_values = portfolio_sims[-1, :]
                st.write("### Distribution of Final Portfolio Values")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.hist(final_values, bins=30, edgecolor="black", alpha=0.7)
                ax2.set_xlabel("Final Portfolio Value ($)")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Distribution of Final Portfolio Values")
                st.pyplot(fig2)

            st.write("#### Simulation and Data Settings")
            st.dataframe(settings_df)

            st.write("#### Results Interpretation")
            st.write(
                f"There is a {risk_level*100:.0f}% chance that the value of your portfolio will fall below "
                f"${float(portfolio_value) - VaR:.2f} in {int(days)} days "
                f"(confidence level {confidence_interval*100:.0f}%)."
            )
            st.write(
                f"With {confidence_interval*100:.0f}% confidence, the potential loss over the next {int(days)} days "
                f"won't exceed ${VaR:.2f} (based on this model)."
            )

        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()
