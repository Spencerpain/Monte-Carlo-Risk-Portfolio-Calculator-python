import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from Value_At_Risk import MonteCarlo, PortfolioOptimizer, Backtester


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def fetch_prices(tickers, start, end):
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
            raise KeyError(f"Could not find price columns. lvl0={sorted(set(lvl0))}")
    else:
        if "Adj Close" in df.columns:
            prices = df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in df.columns:
            prices = df[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise KeyError(f"Could not find price columns: {list(df.columns)}")

    return prices.dropna(how="all").sort_index()


def get_data_local(tickers, start, end):
    prices = fetch_prices(tickers, start, end)

    st.write("Downloaded prices shape:", prices.shape)
    st.write("Price columns:", list(prices.columns))

    all_nan = prices.columns[prices.isna().all()].tolist()
    if all_nan:
        st.warning(f"Dropping tickers with no data: {', '.join(all_nan)}")
        prices = prices.drop(columns=all_nan)

    if prices.shape[1] == 0:
        raise ValueError("No tickers have usable price data.")

    prices = prices.dropna(how="all").ffill()

    if prices.shape[0] < 2:
        raise ValueError("Not enough price rows. Try a longer date range.")

    returns = prices.pct_change().dropna(how="all")

    if returns.empty:
        raise ValueError("Not enough data to compute returns.")

    return returns, returns.mean(), returns.cov()


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Portfolio Risk Calculator",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Monte Carlo Portfolio Risk Calculator")
    st.write("##### Simulate portfolio risk, decompose exposures, optimise weights, and backtest your VaR model.")
    st.markdown("This tool is a **suggestion only** — invest at your own risk.")

    # ── Sidebar ──────────────────────────────
    st.sidebar.title("⚙️ Settings")

    with st.sidebar.expander("📊 Data Settings", expanded=True):
        tickers_raw = st.text_input("Stock Tickers", "SPY, QQQ, SMH, GLD, TLT")
        stock_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

        weights_raw = st.text_input("Portfolio Weights", "0.2, 0.2, 0.2, 0.2, 0.2")
        weights = np.array(
            [float(w.strip()) for w in weights_raw.split(",") if w.strip()], dtype=float
        )

        years_of_data = st.number_input("Years of Historical Data", 1, 30, 20)
        end_date = dt.datetime.now().strftime("%Y-%m-%d")
        start_date = (
            dt.datetime.now() - dt.timedelta(days=int(years_of_data) * 365)
        ).strftime("%Y-%m-%d")

        portfolio_value = st.number_input("Initial Portfolio Value ($)", value=10000.0)

    with st.sidebar.expander("🎲 Simulation Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            days = st.number_input("Days to Simulate", 2, 1260, 252, step=1)
        with c2:
            simulations = st.number_input("Simulations", 100, 10000, 1000, step=100)

        confidence_interval = st.number_input("Confidence Interval", 0.90, 0.99, 0.95)

    risk_level = 1.0 - confidence_interval

    run = st.sidebar.button("▶ Run Simulation")

    # ── Run ──────────────────────────────────
    if run:
        if not stock_list:
            st.error("Enter at least one ticker.")
            st.stop()
        if len(weights) != len(stock_list):
            st.error("Number of weights must match number of tickers.")
            st.stop()
        if not np.isclose(weights.sum(), 1.0, atol=1e-4):
            st.error(f"Weights must sum to 1. Current sum: {weights.sum():.4f}")
            st.stop()

        with st.spinner("Downloading data and running simulation…"):
            returns, mean_returns, cov_matrix = get_data_local(
                stock_list, start=start_date, end=end_date
            )

            # Align weights to actual returned tickers
            actual_tickers = list(returns.columns)
            if len(actual_tickers) != len(weights):
                st.error(
                    f"Got {len(actual_tickers)} tickers after cleaning but {len(weights)} weights."
                )
                st.stop()

            portfolio_sims = MonteCarlo.run_simulation(
                weights=weights,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                portfolio_value=float(portfolio_value),
                days=int(days),
                simulations=int(simulations),
            )

            VaR = MonteCarlo.calculate_var(portfolio_sims, confidence_interval, float(portfolio_value))
            ES = MonteCarlo.calculate_es(portfolio_sims, confidence_interval, float(portfolio_value))
            component_es = MonteCarlo.calculate_component_es(
                weights, returns, confidence_interval, float(portfolio_value)
            )
            risk_decomp = MonteCarlo.calculate_risk_decomposition(
                weights, returns, confidence_interval, float(portfolio_value)
            )

        st.session_state["results"] = dict(
            portfolio_sims=portfolio_sims,
            returns=returns,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            weights=weights,
            stock_list=actual_tickers,
            portfolio_value=float(portfolio_value),
            VaR=VaR,
            ES=ES,
            component_es=component_es,
            risk_decomp=risk_decomp,
            confidence_interval=confidence_interval,
            risk_level=risk_level,
            days=int(days),
            simulations=int(simulations),
            start_date=start_date,
            end_date=end_date,
            years_of_data=int(years_of_data),
        )

    # ── Display ───────────────────────────────
    if "results" not in st.session_state:
        st.info("Configure settings in the sidebar and click **▶ Run Simulation**.")
        return

    r = st.session_state["results"]
    portfolio_sims = r["portfolio_sims"]
    returns = r["returns"]
    mean_returns = r["mean_returns"]
    cov_matrix = r["cov_matrix"]
    weights = r["weights"]
    stock_list = r["stock_list"]
    portfolio_value = r["portfolio_value"]
    VaR = r["VaR"]
    ES = r["ES"]
    component_es = r["component_es"]
    risk_decomp = r["risk_decomp"]
    confidence_interval = r["confidence_interval"]
    risk_level = r["risk_level"]
    days = r["days"]
    simulations = r["simulations"]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Simulation", "🔍 Risk Decomposition", "⚖️ Optimisation", "🧪 Backtest"]
    )

    # ── Tab 1: Simulation ─────────────────────
    with tab1:
        st.subheader("Key Risk Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Value at Risk (VaR)", f"${VaR:,.2f}")
        m2.metric("Expected Shortfall (ES)", f"${ES:,.2f}")
        m3.metric(
            f"Floor Value ({confidence_interval*100:.0f}% CI)",
            f"${portfolio_value - VaR:,.2f}",
        )
        m4.metric("ES / VaR Ratio", f"{ES / VaR:.2f}x" if VaR > 0 else "—")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Simulation Paths")
            fig, ax = plt.subplots(figsize=(6, 4))
            max_paths = min(simulations, 300)
            for i in range(max_paths):
                ax.plot(portfolio_sims[:, i], alpha=0.15, linewidth=0.6, color="steelblue")
            ax.plot(np.mean(portfolio_sims, axis=1), linewidth=2, color="navy", label="Mean Path")
            ax.axhline(
                y=portfolio_value - VaR,
                color="red",
                linewidth=1.5,
                linestyle="--",
                label=f"VaR floor: ${portfolio_value - VaR:,.0f}",
            )
            ax.axhline(
                y=portfolio_value - ES,
                color="orange",
                linewidth=1.5,
                linestyle=":",
                label=f"ES floor: ${portfolio_value - ES:,.0f}",
            )
            ax.set_title(f"{days}-day simulation ({simulations} trials)")
            ax.set_xlabel("Days")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.write("### Final Value Distribution")
            final_values = portfolio_sims[-1, :]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.hist(final_values, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
            ax2.axvline(portfolio_value - VaR, color="red", linewidth=1.5, linestyle="--", label=f"VaR: ${VaR:,.0f}")
            ax2.axvline(portfolio_value - ES, color="orange", linewidth=1.5, linestyle=":", label=f"ES: ${ES:,.0f}")
            ax2.set_xlabel("Final Portfolio Value ($)")
            ax2.set_ylabel("Frequency")
            ax2.legend(fontsize=8)
            st.pyplot(fig2)
            plt.close(fig2)

        st.write("#### Interpretation")
        st.write(
            f"With **{confidence_interval*100:.0f}% confidence**, your portfolio will not lose more than "
            f"**${VaR:,.2f}** over {days} days."
        )
        st.write(
            f"If losses **do** exceed VaR, the average loss is **${ES:,.2f}** (Expected Shortfall). "
            f"ES is the Basel III/IV standard risk measure."
        )

        with st.expander("Simulation Parameters"):
            settings_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Tickers", "Weights", "Years of Data", "Start Date", "End Date",
                        "Portfolio Value", "Days", "Simulations", "Confidence Interval",
                        "VaR", "ES",
                    ],
                    "Value": [
                        ", ".join(stock_list),
                        ", ".join(f"{w:.2f}" for w in weights),
                        r["years_of_data"],
                        r["start_date"], r["end_date"],
                        f"${portfolio_value:,.2f}",
                        days, simulations,
                        f"{confidence_interval*100:.1f}%",
                        f"${VaR:,.2f}", f"${ES:,.2f}",
                    ],
                }
            ).set_index("Parameter")
            st.dataframe(settings_df)

    # ── Tab 2: Risk Decomposition ─────────────
    with tab2:
        st.subheader("Risk Decomposition")
        st.write(
            "Understand which positions are driving your tail risk — "
            "not just total portfolio VaR but each asset's contribution."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Marginal & Component VaR")
            st.dataframe(risk_decomp.style.format({
                "Weight (%)": "{:.1f}%",
                "Marginal VaR (%)": "{:.2f}%",
                "Component VaR ($)": "${:.2f}",
                "Component VaR (%)": "{:.1f}%",
            }), use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(stock_list, risk_decomp["Component VaR (%)"], color="steelblue", edgecolor="black")
            ax.set_ylabel("% of Total VaR")
            ax.set_title("Component VaR Attribution")
            ax.axhline(100 / len(stock_list), color="red", linestyle="--", label="Equal share")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.write("#### Component Expected Shortfall")
            ces_df = component_es.reset_index()
            ces_df.columns = ["Ticker", "Component ES ($)"]
            ces_df["Component ES (%)"] = ces_df["Component ES ($)"] / ces_df["Component ES ($)"].sum() * 100
            st.dataframe(ces_df.set_index("Ticker").style.format({
                "Component ES ($)": "${:.2f}",
                "Component ES (%)": "{:.1f}%",
            }), use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(ces_df["Ticker"], ces_df["Component ES (%)"], color="tomato", edgecolor="black")
            ax.set_ylabel("% of Total ES")
            ax.set_title("Component ES Attribution")
            ax.axhline(100 / len(stock_list), color="navy", linestyle="--", label="Equal share")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        st.info(
            "**How to read this:** If a position has a Component VaR % much larger than its Weight %, "
            "it is contributing disproportionately to tail risk and may warrant a smaller allocation."
        )

    # ── Tab 3: Optimisation ───────────────────
    with tab3:
        st.subheader("Portfolio Optimisation")
        st.write("Compare your current weights against optimised portfolios.")

        with st.spinner("Running optimisers…"):
            w_sharpe = PortfolioOptimizer.max_sharpe(mean_returns, cov_matrix)
            w_minvar = PortfolioOptimizer.min_variance(mean_returns, cov_matrix)
            w_mincvar = PortfolioOptimizer.min_cvar(returns, confidence_interval)
            w_rp = PortfolioOptimizer.risk_parity(cov_matrix)
            frontier_ret, frontier_vol = PortfolioOptimizer.efficient_frontier(mean_returns, cov_matrix)

        def stats(w):
            ret, vol, sharpe = PortfolioOptimizer.portfolio_stats(w, mean_returns, cov_matrix)
            return ret, vol, sharpe

        portfolios = {
            "Current": weights,
            "Max Sharpe": w_sharpe,
            "Min Variance": w_minvar,
            "Min CVaR": w_mincvar,
            "Risk Parity": w_rp,
        }

        summary_rows = []
        for name, w in portfolios.items():
            ret, vol, sharpe = stats(w)
            summary_rows.append({
                "Portfolio": name,
                "Ann. Return": f"{ret:.2%}",
                "Ann. Volatility": f"{vol:.2%}",
                "Sharpe Ratio": f"{sharpe:.2f}",
            })

        st.write("#### Portfolio Comparison")
        st.dataframe(pd.DataFrame(summary_rows).set_index("Portfolio"), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Efficient Frontier")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(frontier_vol * np.sqrt(252), frontier_ret * 252, "b-", linewidth=2, label="Frontier")
            colors = {"Current": "black", "Max Sharpe": "green", "Min Variance": "blue",
                      "Min CVaR": "orange", "Risk Parity": "purple"}
            for name, w in portfolios.items():
                ret, vol, _ = stats(w)
                ax.scatter(vol, ret, zorder=5, label=name, color=colors.get(name, "grey"), s=80)
            ax.set_xlabel("Annual Volatility")
            ax.set_ylabel("Annual Return")
            ax.set_title("Efficient Frontier")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.write("#### Weight Comparison")
            weights_df = pd.DataFrame(
                {name: w for name, w in portfolios.items()},
                index=stock_list,
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            weights_df.T.plot(kind="bar", ax=ax, edgecolor="black")
            ax.set_ylabel("Weight")
            ax.set_title("Weights by Portfolio")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.legend(fontsize=7, loc="upper right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.write("#### Optimised Weights Detail")
        st.dataframe(
            weights_df.style.format("{:.1%}"),
            use_container_width=True,
        )

    # ── Tab 4: Backtest ───────────────────────
    with tab4:
        st.subheader("VaR / ES Backtest")
        st.write(
            "Roll a historical window forward, compute VaR each day, "
            "and check how often actual losses exceeded it (exceedance test)."
        )

        backtest_window = st.slider("Rolling Window (days)", 60, 504, 252, step=21)

        if len(returns) < backtest_window + 10:
            st.warning("Not enough historical data for the selected window. Try a shorter window or more years of data.")
        else:
            with st.spinner("Running backtest…"):
                bt = Backtester.rolling_var_backtest(
                    returns, weights, confidence_interval, window=backtest_window
                )
                kupiec = Backtester.kupiec_test(bt["Exceedance"], confidence_interval)

            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Rolling VaR vs Actual Losses")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(bt.index, bt["VaR"], color="steelblue", linewidth=1, label="1-day VaR")
                ax.plot(bt.index, bt["Actual Loss"], color="grey", linewidth=0.7, alpha=0.6, label="Actual Loss")
                exceedance_dates = bt.index[bt["Exceedance"]]
                ax.scatter(
                    exceedance_dates,
                    bt.loc[exceedance_dates, "Actual Loss"],
                    color="red", s=15, zorder=5, label="Exceedance",
                )
                ax.set_ylabel("Daily P&L (fraction)")
                ax.set_title("VaR Backtest")
                ax.legend(fontsize=8)
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.write("#### Kupiec Traffic Light Test")
                kupiec_df = pd.DataFrame(list(kupiec.items()), columns=["Metric", "Value"]).set_index("Metric")
                st.dataframe(kupiec_df, use_container_width=True)

                zone = kupiec["Traffic Light"]
                if "Green" in zone:
                    st.success(zone)
                elif "Yellow" in zone:
                    st.warning(zone)
                else:
                    st.error(zone)

            st.info(
                "**Kupiec test:** Checks whether the proportion of losses exceeding VaR is "
                "statistically consistent with the confidence level. "
                "Green = model is calibrated well. Red = model is mis-specified."
            )


if __name__ == "__main__":
    main()
