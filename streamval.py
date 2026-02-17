import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from Value_At_Risk import MonteCarlo


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download prices from yfinance and return a DataFrame with columns = tickers.
    Tries 'Adj Close' first, then falls back to 'Close'. Handles MultiIndex outputs.
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise ValueError("No data returned from Yahoo Finance. Check tickers/date range.")

    # MultiIndex case (common for multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        if "Adj Close" in lvl0:
            prices = df["Adj Close"]
        elif "Close" in lvl0:
            prices = df["Close"]
        elif "Adj Close" in lvl1:
            prices = df.xs("Adj Close", axis=1, level=1)
        elif "Close" in lvl1:
            prices = df.xs("Close", axis=1, level=1)
        else:
            raise KeyError(
                f"Could not find 'Adj Close' or 'Close'. "
                f"lvl0={sorted(set(lvl0))}, lvl1={sorted(set(lvl1))}"
            )

    # Single ticker case (single-level columns)
    else:
        if "Adj Close" in df.columns:
            prices = df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in df.columns:
            prices = df[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise KeyError(f"Could not find 'Adj Close' or 'Close'. Columns: {list(df.columns)}")

    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("Prices are empty after dropping missing rows.")

    return prices


def get_data_local(tickers: list[str], start: str, end: str):
    """
    Returns: returns, mean_returns, cov_matrix
    """
    prices = fetch_prices(tickers, start, end)
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough data to compute returns. Try a longer date range.")
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return returns, mean_returns, cov_matrix


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

        portfolio_value = st.number_input(
            "Initial Portfolio Value ($):",
            value=10000.0,
            help="Account balance",
        )

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

    if st.sidebar.button("Run Simulation"):
        try:
            if len(stock_list) == 0:
                st.error("Please enter at least one ticker.")
                st.stop()

            if len(weights) != len(stock_list):
                st.error("Number of weights must match number of tickers.")
                st.stop()

            if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                st.error(f"Weights must sum to 1. Current sum: {weights.sum():.6f}")
                st.stop()

            returns, mean_returns, cov_matrix = get_data_local(stock_list, start=start_date, end=end_date)

            portfolio_sims = MonteCarlo.run_simulation(
                weights=weights,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                portfolio_value=float(portfolio_value),
                days=int(days),
                simulations=int(simulations),
            )

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

            with plot_col2:
                st.write("### Distribution of Final Portfolio Values")
                final_values = portfolio_sims[-1, :]
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
