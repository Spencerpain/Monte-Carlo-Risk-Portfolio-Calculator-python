import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import streamlit as st

from Get_Data import get_data
from Value_At_Risk import MonteCarlo


def main():
    # PAGE SETTINGS
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

    # SIDEBAR SETTINGS
    st.sidebar.title("⚙️ Settings")

    # Expander for Data Settings
    with st.sidebar.expander("📊 Data Settings", expanded=True):
        # Input: Stock tickers (robust parsing)
        tickers_raw = st.text_input(
            "Stock Tickers",
            "SPY, QQQ, SMH, GLD, TLT",
            help="Enter stock tickers separated by commas (e.g., SPY, QQQ, SMH, GLD, TLT)",
        )
        stock_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

        # Input: Weights
        weights_input = st.text_input(
            "Portfolio Weights",
            "0.2, 0.2, 0.2, 0.2, 0.2",
            help="Enter portfolio weights separated by commas (must sum to 1).",
        )
        weights = np.array([float(w.strip()) for w in weights_input.split(",") if w.strip()], dtype=float)

        # Input: Number of years for historical data
        years_of_data = st.number_input(
            "Years of Historical Data",
            min_value=1,
            max_value=30,
            value=20,
            help="Select number of years to include in historical data (most recent n years)",
        )

        end_date = dt.datetime.now().strftime("%Y-%m-%d")
        start_date = (dt.datetime.now() - dt.timedelta(days=int(years_of_data) * 365)).strftime("%Y-%m-%d")

        # Input: Portfolio initial value
        portfolio_value = st.number_input(
            "Initial Portfolio Value ($):",
            value=10000.0,
            help="Account balance",
        )

    # Expander for Simulation Settings
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
            help="Select confidence interval for VaR (0.90 to 0.99)",
        )

    risk_level = 1.0 - confidence_interval

    # RUN SIMULATION
    if st.sidebar.button("Run Simulation"):
        try:
            # Basic validation
            if len(stock_list) == 0:
                st.error("Please enter at least one ticker.")
                st.stop()

            if len(weights) != len(stock_list):
                st.error("Number of weights must match number of tickers.")
                st.stop()

            if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                st.error(f"Weights must sum to 1. Current sum: {weights.sum():.6f}")
                st.stop()

            # Fetch data
            returns, mean_returns, cov_matrix = get_data(stock_list, start=start_date, end=end_date)

            # Run Monte Carlo simulation (uses your Value_At_Risk.py implementation)
            portfolio_sims = MonteCarlo.run_simulation(
                weights=weights,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                portfolio_value=float(portfolio_value),
                days=int(days),
                simulations=int(simulations),
            )

            # Calculate VaR
            VaR = MonteCarlo.calculate_var(portfolio_sims, confidence_interval, float(portfolio_value))

            # Settings table
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

            # Layout for plots
            plot_col1, plot_col2 = st.columns(2)

            # Plot simulation results
            with plot_col1:
                st.write("### Simulation Results")
                fig, ax = plt.subplots(figsize=(6, 4))

                # Plot a capped number of paths for speed/clarity
                max_paths_to_plot = min(int(simulations), 300)
                for i in range(max_paths_to_plot):
                    ax.plot(portfolio_sims[:, i], alpha=0.20, linewidth=0.7)

                mean_path = np.mean(portfolio_sims, axis=1)
                ax.plot(mean_path, linewidth=2, label="Mean Simulation")

                ax.axhline(
                    y=(float(portfolio_value) - VaR),
                    linewidth=1,
                    label=f"VaR ({confidence_interval*100:.0f}%): ${VaR:.2f}",
                )

                ax.set_title(f"Portfolio Value over {int(days)} days ({int(simulations)} trials)")
                ax.set_xlabel("Days")
                ax.set_ylabel("Portfolio Value ($)")
                ax.legend(loc="upper left")
                st.pyplot(fig)

            # Plot histogram of final portfolio values
            with plot_col2:
                st.write("### Distribution of Final Portfolio Values")
                final_values = portfolio_sims[-1, :]
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.hist(final_values, bins=30, edgecolor="black", alpha=0.7)
                ax2.set_xlabel("Final Portfolio Value ($)")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Distribution of Final Portfolio Values")
                st.pyplot(fig2)

            # Display settings + interpretation
            st.write("#### Simulation and Data Settings")
            st.dataframe(settings_df)

            st.write("#### Results Interpretation")
            st.write(
                f"There is a {risk_level*100:.0f}% chance your portfolio will fall below "
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
