📈 AI Portfolio Optimization System
An intelligent portfolio optimization system built using Machine Learning, XGBoost, and Modern Portfolio Theory to help investors allocate assets efficiently based on risk preferences.

🚀 Features
✅ Historical stock & crypto data collection using Yahoo Finance
✅ Portfolio optimization using PyPortfolioOpt
✅ AI-based future return prediction using XGBoost
✅ Hybrid return modeling (Historical + AI Predictions)
✅ Risk profiling:

Conservative

Moderate

Aggressive

✅ Interactive Streamlit Dashboard
✅ Portfolio allocation visualization
✅ Investment breakdown calculator
✅ Sharpe Ratio optimization
✅ Actual vs Predicted return comparison
✅ Download portfolio report feature

🧠 Technologies Used
Programming Language
Python

Libraries & Frameworks
pandas

numpy

matplotlib

plotly

seaborn

yfinance

scikit-learn

xgboost

PyPortfolioOpt

streamlit

📊 Machine Learning Model
The project uses XGBoost Regressor for predicting future market returns.

Features Used
Moving Average (5-day)

Moving Average (10-day)

Volatility

Evaluation Metrics
R² Score

Mean Squared Error (MSE)

📂 Project Structure
AI-Portfolio-Optimizer/
│
├── app.py
├── main.ipynb
├── requirements.txt
├── portfolio_data.csv
│
├── README.md
│
└── screenshots/
⚙ Installation
1. Clone Repository
git clone https://github.com/your-username/AI-Portfolio-Optimizer.git
2. Install Dependencies
pip install -r requirements.txt
▶ Running the Application
Run the Streamlit dashboard:

streamlit run app.py
📈 Dashboard Features
🔹 Asset Selection
Choose multiple assets like:

Apple

Microsoft

Google

Amazon

Gold ETF

Bitcoin

🔹 Return Models
Users can choose between:

Historical Returns

XGBoost AI Predictions

Hybrid Model

🔹 Risk Profiles
Conservative → Minimum volatility

Moderate → Maximum Sharpe ratio

Aggressive → Maximum utility portfolio

🔹 Visualizations
Historical Price Trends

Pie Charts

Allocation Bar Charts

Actual vs Predicted Returns

📌 Portfolio Optimization Strategy
The project uses:

Mean-Variance Optimization

Efficient Frontier

Sharpe Ratio Maximization

to determine optimal asset allocation.

🔮 Future Improvements
LSTM-based deep learning prediction

Real-time stock market integration

News sentiment analysis

AI chatbot financial advisor

Dark mode UI

Live portfolio tracking

📷 Sample Output
Portfolio Allocation
AAPL      35%
MSFT      30%
GLD       20%
BTC       15%
🎯 Learning Outcomes
This project demonstrates:

Financial data analysis

Machine learning for finance

Portfolio optimization

Risk analysis

Data visualization

Streamlit web development
