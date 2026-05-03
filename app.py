# =========================================================
# AI PORTFOLIO OPTIMIZER - STREAMLIT UI
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📈 AI Portfolio Optimization System")
st.markdown("### Smart Investment Allocation using ML & Optimization")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙ User Inputs")

investment = st.sidebar.number_input(
    "Investment Amount ($)",
    min_value=100,
    value=10000
)

risk_level = st.sidebar.selectbox(
    "Select Risk Profile",
    ["Conservative", "Moderate", "Aggressive"]
)

assets = st.sidebar.multiselect(
    "Choose Assets",
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'GLD', 'BTC-USD'],
    default=['AAPL', 'MSFT', 'GLD']
)

# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------

model_choice = st.sidebar.selectbox(
    "Select Return Model",
    ["Historical", "XGBoost", "Hybrid"]
)

# ---------------------------------------------------------
# DOWNLOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data(assets):

    data = yf.download(
        assets,
        start='2020-01-01',
        end='2025-01-01'
    )['Close']

    return data

data = load_data(assets)

# ---------------------------------------------------------
# SHOW DATA
# ---------------------------------------------------------

st.subheader("📊 Historical Price Data")
st.dataframe(data.tail())

# ---------------------------------------------------------
# PRICE CHART
# ---------------------------------------------------------

st.subheader("📉 Asset Price Trends")

fig = px.line(
    data,
    x=data.index,
    y=data.columns,
    title="Historical Asset Prices"
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# RETURNS
# =========================================================

returns = data.pct_change().dropna()

# =========================================================
# XGBOOST SECTION
# =========================================================

returns['Market_Return'] = returns.mean(axis=1)

returns['MA_5'] = returns['Market_Return'].rolling(5).mean()
returns['MA_10'] = returns['Market_Return'].rolling(10).mean()
returns['Volatility'] = returns['Market_Return'].rolling(10).std()

returns = returns.dropna()

X = returns[['MA_5', 'MA_10', 'Volatility']]
y = returns['Market_Return']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ---------------------------------------------------------
# XGBOOST MODEL
# ---------------------------------------------------------

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# ---------------------------------------------------------
# RETURN MODEL SELECTION
# ---------------------------------------------------------

historical_mu = expected_returns.mean_historical_return(data)

predicted_return = np.mean(predictions)

xgb_mu = pd.Series(
    [predicted_return] * len(data.columns),
    index=data.columns
)

if model_choice == "Historical":
    mu = historical_mu

elif model_choice == "XGBoost":
    mu = xgb_mu

else:
    mu = (historical_mu + xgb_mu) / 2

# =========================================================
# PORTFOLIO OPTIMIZATION
# =========================================================

S = risk_models.sample_cov(data)

ef = EfficientFrontier(mu, S)

if risk_level == "Conservative":
    ef.min_volatility()

elif risk_level == "Moderate":
    ef.max_sharpe()

else:
    ef.max_quadratic_utility()

weights = ef.clean_weights()

# =========================================================
# MODEL INFO
# =========================================================

st.subheader("🧠 Return Prediction Model")

st.write(f"### Selected Model: {model_choice}")

if model_choice == "Historical":
    st.info("Using historical market returns")

elif model_choice == "XGBoost":
    st.warning("Using AI predicted returns from XGBoost")

else:
    st.success("Using Hybrid AI + Historical returns")

# =========================================================
# XGBOOST PREDICTION GRAPH
# =========================================================

st.subheader("🔮 XGBoost Future Return Prediction")

prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

fig_pred = px.line(
    prediction_df,
    title="Actual vs Predicted Returns"
)

st.plotly_chart(fig_pred, use_container_width=True)

# =========================================================
# ML METRICS
# =========================================================

col4, col5 = st.columns(2)

col4.metric(
    "R² Score",
    f"{r2:.4f}"
)

col5.metric(
    "Mean Squared Error",
    f"{mse:.6f}"
)

# =========================================================
# WEIGHTS
# =========================================================

st.subheader("💼 Optimal Portfolio Allocation")

weights_df = pd.DataFrame(
    list(weights.items()),
    columns=['Asset', 'Weight']
)

weights_df['Weight'] = weights_df['Weight'] * 100

# ---------------------------------------------------------
# PIE CHART
# ---------------------------------------------------------

fig2 = px.pie(
    weights_df,
    names='Asset',
    values='Weight',
    title='Portfolio Distribution'
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# BAR CHART
# ---------------------------------------------------------

fig3 = px.bar(
    weights_df,
    x='Asset',
    y='Weight',
    color='Asset',
    title='Asset Allocation (%)'
)

st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# INVESTMENT BREAKDOWN
# =========================================================

st.subheader("💰 Investment Breakdown")

weights_df['Investment'] = (
    weights_df['Weight'] / 100
) * investment

st.dataframe(weights_df)

# =========================================================
# PERFORMANCE METRICS
# =========================================================

st.subheader("📌 Portfolio Performance")

expected_return, volatility, sharpe = ef.portfolio_performance()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Expected Annual Return",
    f"{expected_return*100:.2f}%"
)

col2.metric(
    "Annual Volatility",
    f"{volatility*100:.2f}%"
)

col3.metric(
    "Sharpe Ratio",
    f"{sharpe:.2f}"
)

# =========================================================
# RISK MESSAGE
# =========================================================

if risk_level == "Conservative":
    st.success("Low Risk Portfolio Selected")

elif risk_level == "Moderate":
    st.info("Balanced Risk-Reward Portfolio")

else:
    st.warning("High Risk High Reward Portfolio")

# =========================================================
# DOWNLOAD BUTTON
# =========================================================

csv = weights_df.to_csv(index=False)

st.download_button(
    label="📥 Download Portfolio Report",
    data=csv,
    file_name='portfolio_report.csv',
    mime='text/csv'
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit, XGBoost & PyPortfolioOpt"
)