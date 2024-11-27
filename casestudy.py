from yahooquery import Ticker
from scipy.stats import norm
import pandas as pd
import numpy as np
import argparse

# Parse command line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Calculate Black-Scholes value of call options")
    parser.add_argument("--stock", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--risk_free_rate", type=float, default=0.045, help="Risk-free rate (default: 0.045)")

    return parser.parse_args()

# Define Black-Scholes formula for call options
def black_scholes_call(S, K, T, r, sigma, q):
    '''
    S: Current asset price
    K: Option strike price
    T: Time to expiration
    r: Risk-free rate
    sigma: Annualized volatility of asset returns
    q: Dividend yield
    '''
    try:
        d1 = (np.log(S/K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - norm.cdf(d2) * K * np.exp(-r * T)
        return call_price
    except:
        return 'n/a'

# Main
def main():
    args = get_args()
    stock = args.stock
    risk_free_rate = args.risk_free_rate

    # Import options data via YahooQuery
    ticker = Ticker(stock)
    options = ticker.option_chain
    options_df = options.reset_index()
    calls_df = options_df[options_df['optionType'] == 'calls']

    # Add mid_price column and filter by > 1
    calls_df.loc[:, 'mid_price'] = (calls_df['ask'] + calls_df['bid']) / 2
    calls_df = calls_df[calls_df['mid_price'] > 1]

    # Fetch stock price
    current_price = ticker.price[stock]['regularMarketPrice']

    # Fetch dividend yield
    dividend_yield = ticker.summary_detail[stock].get('dividendYield')
    dividend_yield = dividend_yield if dividend_yield is not None else 0 # In case null version of dividend yield is 0

    # Calculate days to expiration
    calls_df['daysToExpiration'] = (calls_df['expiration'] - pd.Timestamp.today()).dt.days

    # Compute call option price using Black-Scholes formula
    calls_df['black_scholes_value'] = calls_df.apply(
        lambda row: black_scholes_call(current_price, row['strike'], row['daysToExpiration'] / 365, risk_free_rate, dividend_yield, row['impliedVolatility'])
        , axis=1
    )

    # Calculate price difference
    calls_df["difference_in_price"] = calls_df["mid_price"] - calls_df["black_scholes_value"]

    # Output to CSV
    calls_df = calls_df[['symbol', 'contractSymbol', 'expiration', 'strike', 'bid', 'ask', 'impliedVolatility',  'mid_price', 'daysToExpiration', 'black_scholes_value', 'difference_in_price']]
    calls_df.to_csv(f"{stock}_call_options_values.csv", index=False)
    print("Output saved to f{stock}call_options_values.csv")


if __name__ == "__main__":
    main()