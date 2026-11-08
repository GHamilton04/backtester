import pandas as pd

def sma_crossover(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Simple Moving Average (SMA) Crossover Strategy.
    
    Generates buy/sell signals based on the crossover of two SMAs. 
    
    Each signal uses that days closing price, so it must be shifted before trading on it.

    Parameters:
    - prices: A pandas Series of closing prices.
    - fast: The period for the fast SMA.
    - slow: The period for the slow SMA.
    
    Returns:
    - A pandas Series with buy/sell signals (1 for holding shares, 0 for holding nothing).
    """
    if len(prices) < slow:
        raise ValueError("Invalid input: The length of the price series must be greater than or equal to the slow SMA period.")

    if fast >= slow:
        raise ValueError(f"Invalid input: The fast SMA period ({fast}) must be less than the slow SMA period ({slow}).")

    if fast <= 0 or slow <= 0:
        raise ValueError(f"Invalid input: SMA periods ({fast}, {slow}) must be positive integers.")

    fast_sma = prices.rolling(window=fast).mean()
    slow_sma = prices.rolling(window=slow).mean()

    signals = pd.Series(0, index=prices.index)
    signals[fast_sma > slow_sma] = 1

    return signals