from matplotlib.pyplot import spy
import pandas as pd


def run_backtest(prices: pd.Series, signals: pd.Series, cost_rate: float = 0.001, initial_capital: float = 10_000,) -> pd.DataFrame:
    """
    Backtest of a trading strategy given daily prices and signals.

    Returns a DataFrame with the strategy results over time accounting for transaction costs, 
    against a benchmark buy-and-hold strategy.

    Signals should be unshifted, because the function shifts them itself
    
    Each change is treated as a trade at that day's close.

    Assumptions:
    cash earns nothing and that the benchmark has no costs.

    Parameters:
    prices : pd.Series
        Daily adjusted closing prices, sorted oldest first, with no missing values.
    signals : pd.Series
        1 or 0 for each date in prices, with the same index.
    cost_rate : float, default 0.001
        Transaction cost rate per trade, as a decimal (e.g., 0.001 for 0.1%).
    initial_capital : float, default 10_000
        Starting capital for the backtest.
    
    Returns:
    pd.DataFrame
        A DataFrame with the following columns:
        - price: daily adjusted closing prices
        - signal: trading signals (1 or 0)
        - position: whether shares are held (0 = no, 1 = yes)
        - market: daily market returns
        - strategy: daily strategy returns
        - trades: number of trades executed (0 or 1)
        - cost: transaction costs incurred
        - net: net returns after costs
        - equity: cumulative equity over time
        - benchmark: cumulative benchmark returns over time

    Raises:
    ValueError
        - if prices and signals' dates do not match.
        - if signals contain values other than 0 or 1.
        - if cost_rate is negative.
        - if initial_capital is not positive.    
    
    """

    if not prices.index.equals(signals.index):
        raise ValueError("Invalid input: Prices and Signals' dates do not match.")
    
    if not signals.isin([0, 1]).all():
        raise ValueError("Invalid input: Signals must be 0 or 1.")

    if cost_rate < 0:
        raise ValueError(f"Invalid input: The cost rate ({cost_rate}) must be non-negative.")

    if initial_capital <= 0:
        raise ValueError(f"Invalid input: The initial capital ({initial_capital}) must be positive.")

    market_returns = prices.pct_change().fillna(0)
    positions = signals.shift(1).fillna(0).astype(int)
    strategy_returns = market_returns * positions

    trades = positions.diff().abs().fillna(0)
    cost = trades * cost_rate
    net_return = strategy_returns - cost

    equity = initial_capital * (1 + net_return).cumprod()
    benchmark = initial_capital * (1 + market_returns).cumprod()

    results = pd.DataFrame({
        "price": prices,
        "signal": signals,
        "position": positions,
        "market": market_returns,
        "strategy": strategy_returns,
        "trades": trades,
        "cost": cost,
        "net": net_return,
        "equity": equity,
        "benchmark": benchmark,
    })

    return results


def extract_trades(prices: pd.Series, signals: pd.Series, cost_rate: float = 0.001,) -> pd.DataFrame:
    """
    Extract trade information from a series of prices and signals.

    Returns a DataFrame with the entry and exit dates, prices, days held, and net return for each trade.

    An open trade at the end is closed at the last price.

    Parameters:
    prices : pd.Series
        Daily adjusted closing prices, sorted oldest first, with no missing values.
    signals : pd.Series
        1 or 0 for each date in prices, with the same index.
    cost_rate : float, default 0.001
        Transaction cost rate per trade, as a decimal (e.g., 0.001 for 0.1%).
    
    Returns:
    pd.DataFrame
        A DataFrame with the following columns:
        - entry_date: date of trade entry
        - exit_date: date of trade exit
        - entry_price: price at trade entry
        - exit_price: price at trade exit
        - days_held: number of days the trade was held (calendar days)
        - net_return: net return of the trade after costs

    Raises:
    ValueError
        - if prices and signals' dates do not match.
        - if signals contain values other than 0 or 1.
    """

    if not prices.index.equals(signals.index):
        raise ValueError("Invalid input: Prices and Signals' dates do not match.")
        
    if not signals.isin([0, 1]).all():
        raise ValueError("Invalid input: Signals must be 0 or 1.")

    changes = signals.diff()
    entries = list(signals.index[changes == 1])
    exits = list(signals.index[changes == -1])

    if len(entries) > len(exits):
        exits.append(signals.index[-1])
    
    trade_log = pd.DataFrame({"entry_date": entries, "exit_date": exits})
    trade_log["entry_price"] = prices.loc[entries].to_numpy()
    trade_log["exit_price"] = prices.loc[exits].to_numpy()
    trade_log["days_held"] = (trade_log["exit_date"] - trade_log["entry_date"]).dt.days
    trade_log["net_return"] = trade_log["exit_price"] / trade_log["entry_price"] - 1 - 2 * cost_rate

    return trade_log