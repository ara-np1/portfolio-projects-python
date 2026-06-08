# portfolio-monte-carlo
Monte Carlo portfolio simulator (READ python_mc.py)
# Portfolio Monte Carlo Simulator

A Python tool simulating 5-year performance of a diversified global 
portfolio across 16 assets using Monte Carlo methods.

## What it does
- Downloads live price data via yfinance
- Runs 10,000 simulations using log-normal return modelling
- Computes VaR, CVaR, Sharpe Ratio and Sortino Ratio
- Outputs a distribution histogram with annotated risk metrics

## Technologies
Python · NumPy · Matplotlib · yfinance

## How to Run
pip install numpy matplotlib yfinance
python monte_carlo.py

## Improvements
Fix the inflated Sharpe Ratio and Sortino Ratio
Rebalance & improve diversification




# Nederlands
De meeste berekeningen voor later gaan uit van een simpel lijntje omhoog, maar de beurs werkt niet zo. Om een eerlijker beeld te krijgen, heb ik een tool gebouwd die tienduizend verschillende scenario's doorrekent voor een mix van zestien assets. De basis hiervoor is live data; via yfinance haalt de code de meest recente koersen op en berekent zelf hoe hard deze assets schommelen. Zo ben je niet afhankelijk van verouderde lijstjes, maar zie je de markt van vandaag.

Een belangrijk detail is hoe de assets onderling op elkaar reageren. In een echte crisis zie je vaak dat alles tegelijkertijd omlaag dondert. Om dit effect mee te nemen, berekent de code een correlatiematrix en past daar een Cholesky-decompositie op toe. Dit dwingt de simulatie om rekening te houden met het feit dat aandelen elkaar beïnvloeden, wat een veel eerlijker beeld geeft van het risico dan wanneer je elk aandeel als een los eilandje zou zien.

Voor de simulaties zelf gebruik ik de Geometrische Brownse Beweging (GBM). Dit model zorgt ervoor dat de berekeningen logisch blijven en niet door een soort sneeuwbaleffect op onmogelijke bedragen uitkomen. Al die tienduizend verschillende jaarlijkse schommelingen worden uiteindelijk met np.cumprod aan elkaar geknoopt tot tienduizend mogelijke eindcijfers.

Om alle data bruikbaar te maken, kijkt de code naar de Value at Risk (VaR) en de Sortino-ratio. De VaR laat zien wat er gebeurt in de slechtste 5% van de scenario's, terwijl de Sortino-ratio alleen kijkt naar de schommelingen waar je écht geld door verliest. Het resultaat is een histogram dat in één oogopslag de kans op verlies en de meest waarschijnlijke uitkomst laat zien.

# English
Most financial projections rely on a simple straight line, but the market doesn't work that way. To get a more honest outlook, I built a tool that runs ten thousand different scenarios for a mix of sixteen assets. It is powered by live data; using yfinance, the code grabs the latest prices and figures out the volatility on its own. This keeps the model tied to what is happening right now, rather than relying on static assumptions.

A key detail is how these assets react to one another. During a real crash, everything tends to drop at the same time. To capture this, the code builds a correlation matrix and applies a Cholesky Decomposition. This forces the simulation to acknowledge that assets affect each other, providing a much more realistic look at risk than treating every stock as if it lives in a vacuum.

To plot out the future, I used Geometric Brownian Motion (GBM). This math ensures that the returns stay realistic and do not spiral into impossible numbers over time. All those thousands of annual ups and downs are linked together using np.cumprod to create ten thousand potential final outcomes for the portfolio.

To make sense of all that data, the code focuses on the Value at Risk (VaR) and the Sortino Ratio. The VaR points out the potential floor by looking at the worst 5% of cases, while the Sortino Ratio ignores "good" volatility and only looks at the drops that actually cost you money. The final result is a histogram that visualizes the likelihood of loss and the most probable outcome at a single glance.
