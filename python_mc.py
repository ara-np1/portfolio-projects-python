import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

allocations = {
    "VWCE.DE": 0.3577,
    "VUAA.L":  0.0988,
    "MVOL.L":  0.0846,
    "VAGF.DE": 0.0694,
    "WQDV.L":  0.0529,
    "4GLD.DE": 0.0522,
    "VHYL.L":  0.0451,
    "EIMI.L":  0.0445,
    "IWDP.L":  0.0418,
    "LMT":     0.0307,
    "JNJ":     0.0292,
    "JPM":     0.0292,
    "ASML":    0.0160,
    "AAPL":    0.0160,
    "NVDA":    0.0160,
    "MSFT":    0.0160,
}

num_simulations    = 10000
time_horizon       = 5
initial_investment = 944.34
risk_free_rate     = 0.03
lookback_years     = 5
np.random.seed(0)

def get_stock_data(tickers, lookback_years):
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=lookback_years * 365)

    print("Downloading stock data...")

    data = yf.download(
        tickers     = list(tickers),
        start       = start_date.strftime("%Y-%m-%d"),
        end         = end_date.strftime("%Y-%m-%d"),
        auto_adjust = True,
        progress    = False
    )["Close"]

    results = {}

    for ticker in tickers:
        try:
            prices            = data[ticker].dropna()
            daily_returns     = prices.pct_change().dropna()
            yearly_return     = (1 + daily_returns.mean()) ** 252 - 1
            yearly_volatility = daily_returns.std() * np.sqrt(252)
            yearly_return     = float(np.clip(yearly_return,     -0.30, 0.40))
            yearly_volatility = float(np.clip(yearly_volatility,  0.05, 0.60))

            results[ticker] = {
                "return":     round(yearly_return,     4),
                "volatility": round(yearly_volatility, 4),
            }

            print(f"  {ticker:10s}  return: {yearly_return*100:5.1f}%   volatility: {yearly_volatility*100:5.1f}%")

        except:
            print(f"  {ticker:10s}  could not load — using default values")
            results[ticker] = {"return": 0.07, "volatility": 0.15}

    print("")
    return results

stock_data  = get_stock_data(allocations.keys(), lookback_years)
assets      = list(allocations.keys())
alloc_list  = np.array([allocations[a]              for a in assets])
return_list = np.array([stock_data[a]["return"]     for a in assets])
vol_list    = np.array([stock_data[a]["volatility"] for a in assets])

if abs(alloc_list.sum() - 1) > 0.001:
    raise ValueError("Allocations must add up to 1.0")

mu = np.log(1 + return_list) - 0.5 * vol_list ** 2

n = len(assets)
corr = np.eye(n)

stocks = {"AAPL", "NVDA", "MSFT", "LMT", "JPM", "JNJ", "ASML"}
etfs   = {"VWCE.DE", "VUAA.L", "MVOL.L", "WQDV.L", "VHYL.L", "EIMI.L", "IWDP.L"}
bonds  = {"VAGF.DE"}
gold   = {"4GLD.DE"}

for i, a1 in enumerate(assets):
    for j, a2 in enumerate(assets):
        if i == j:
            continue

        if a1 in stocks and a2 in stocks:
            corr[i, j] = 0.45
            continue

        if a1 in etfs and a2 in etfs:
            corr[i, j] = 0.40
            continue

        if a1 in bonds or a2 in bonds:
            corr[i, j] = 0.15
            continue

        if a1 in gold or a2 in gold:
            corr[i, j] = -0.05
            continue
        corr[i, j] = 0.20

cov = np.outer(vol_list, vol_list) * corr

L = np.linalg.cholesky(cov)

rand_normals = np.random.normal(size=(num_simulations, time_horizon, n))
correlated_normals = rand_normals @ L.T

random_returns = np.exp(mu + correlated_normals) - 1


portfolio_returns = np.sum(random_returns * alloc_list, axis=2)
portfolio_values  = initial_investment * np.cumprod(1 + portfolio_returns, axis=1)
final_values      = portfolio_values[:, -1]

def calc_var(values, confidence=0.95):
    return np.percentile(values, (1 - confidence) * 100)

def calc_cvar(values, confidence=0.95):
    var = calc_var(values, confidence)
    return values[values <= var].mean()

def calc_sharpe(yearly_returns, risk_free):
    all_years = yearly_returns.flatten()
    excess    = all_years - risk_free
    return excess.mean() / excess.std()

def calc_sortino(yearly_returns, risk_free):
    all_years = yearly_returns.flatten()
    excess    = all_years - risk_free
    bad_years = excess[excess < 0]
    if len(bad_years) < 30:
        return float("nan")
    return excess.mean() / bad_years.std()

median_val  = np.median(final_values)
var_value   = calc_var(final_values)
cvar_value  = calc_cvar(final_values)
sharpe      = calc_sharpe(portfolio_returns, risk_free_rate)
sortino     = calc_sortino(portfolio_returns, risk_free_rate)
prob_loss   = (final_values < initial_investment).mean()
prob_double = (final_values >= 2 * initial_investment).mean()

print(f"Median final value:      ${median_val:,.2f}")
print(f"Value at Risk (95%):     ${var_value:,.2f}")
print(f"Conditional VaR (95%):   ${cvar_value:,.2f}")
print(f"Sharpe Ratio:            {sharpe:.2f}")
print(f"Sortino Ratio:           {sortino:.2f}" if not np.isnan(sortino) else "Sortino Ratio:           N/A")
print(f"Probability of loss:     {prob_loss:.1%}")
print(f"Probability of 2x:       {prob_double:.1%}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(final_values, bins=60, color="steelblue", edgecolor="black", alpha=0.85)

ax.axvline(var_value,          color="red",   linestyle="dashed", linewidth=2)
ax.axvline(median_val,         color="green", linestyle="dashed", linewidth=2)
ax.axvline(initial_investment, color="blue",  linestyle="dashed", linewidth=2)

ymax = ax.get_ylim()[1]
box  = dict(boxstyle="round,pad=0.3", alpha=0.7)

ax.text(var_value,          ymax * 0.92, f"VaR 95%: ${var_value:,.0f}",   color="red",   fontsize=9, bbox=box)
ax.text(median_val,         ymax * 0.80, f"Median: ${median_val:,.0f}",    color="green", fontsize=9, bbox=box)
ax.text(initial_investment, ymax * 0.68, f"Start: ${initial_investment}",  color="blue",  fontsize=9, bbox=box)

today = datetime.today().strftime("%d %b %Y")
ax.set_title(f"Portfolio Simulation — {time_horizon} Years  |  {today}", fontsize=12)
ax.set_xlabel("Portfolio Value ($)")
ax.set_ylabel("Number of simulations")

sortino_str = f"{sortino:.2f}" if not np.isnan(sortino) else "N/A"
summary = (
    f"Sharpe:   {sharpe:.2f}\n"
    f"Sortino:  {sortino_str}\n"
    f"CVaR:    ${cvar_value:,.0f}\n"
    f"P(loss):  {prob_loss:.1%}\n"
    f"P(2x):    {prob_double:.1%}"
)
ax.text(0.98, 0.97, summary, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("monte_carlo_output.png", dpi=150, bbox_inches="tight")
plt.show()
