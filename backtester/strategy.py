import pandas as pd

def sma_crossover(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """    
    Simple moving average (SMA) crossover signal.

    Returns 1 (hold shares) on days when the fast SMA is above the slow SMA,
    and 0 (hold no shares) otherwise.

    Each day's signal uses that day's closing price, so shift it forward one
    day before trading on it, or the backtest will have look-ahead bias.

    The first slow - 1 rows are 0 because the slow SMA can't be calculated
    yet. Treat them as "no signal yet", not as a decision from the SMAs.

    Parameters:
    prices : pd.Series
        Daily adjusted closing prices, sorted oldest first, with no missing
        values.
    fast : int, default 20
        Window for the fast SMA, in trading days.
    slow : int, default 50
        Window for the slow SMA, in trading days. Must be greater than fast.

    Returns:
    pd.Series
        1 or 0 for each date in prices, with the same index.

    Raises:
    ValueError
        If either window isn't positive, if fast isn't less than slow, or if
        prices has fewer rows than slow.
    """
    
    if len(prices) < slow:
        raise ValueError("Invalid input: The length of the price series must be greater than or equal to the slow SMA period.")
    
    if fast <= 0 or slow <= 0:
        raise ValueError(f"Invalid input: SMA periods ({fast}, {slow}) must be positive integers.")

    if fast >= slow:
        raise ValueError(f"Invalid input: The fast SMA period ({fast}) must be less than the slow SMA period ({slow}).")

    fast_sma = prices.rolling(window=fast).mean()
    slow_sma = prices.rolling(window=slow).mean()

    signals = pd.Series(0, index=prices.index)
    signals[fast_sma > slow_sma] = 1

    return signals