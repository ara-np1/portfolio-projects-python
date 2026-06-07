import yfinance as yf
import numpy as np

data = yf.download("JPM", start="2016-02-08", end="2026-02-08", auto_adjust=True)['Close']
returns = data.pct_change().dropna()

daily_return = float(returns.mean().iloc[0])
annual_return = daily_return * 252

daily_vol = float(returns.std().iloc[0]) * 100
annual_vol = float(returns.std().iloc[0]) * np.sqrt(252) * 100

print(f"Expected Daily Return: {daily_return:.4f}")
print(f"Expected Annual Return: {annual_return:.2%}")
print(f"Daily Volatility: {daily_vol:.2f}%")
print(f"Annual Volatility: {annual_vol:.2f}%")
print ("These are the daily and annual returns. This model is also showing the annual and daily volatility.")
print ("Dit zijn de dagelijkse en jaarlijkse rendementen. Dit model toont ook de jaarlijkse en dagelijkse volatiliteit.")
