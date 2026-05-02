import numpy as np

def calculate_daily_net_profit_margin(initial_assets: float, final_assets: float) -> float:
    """
    Calculates the daily net profit margin (P_net) based on Eq. 10.
    P_net = (A_settle - A_initial) / A_initial

    Args:
        initial_assets (float): Total assets at the start of the day (A_initial).
        final_assets (float): Total assets after settlement at the day's close (A_settle).

    Returns:
        float: The daily net profit margin. Returns 0.0 if initial_assets is zero.
    """
    if initial_assets == 0.0:
        # Avoid division by zero. If initial assets are zero, profit margin is undefined or can be considered 0.
        # For cases where final_assets > 0 and initial_assets = 0, it could be considered infinite profit, 
        # but 0.0 is a safer default for metrics that might be averaged.
        return 0.0 
    return (final_assets - initial_assets) / initial_assets

def calculate_win_ratio(winning_trades: int, total_trades: int) -> float:
    """
    Calculates the win ratio (W) based on Eq. 11.
    W = T_w / T_t

    Args:
        winning_trades (int): Number of winning trades (T_w).
        total_trades (int): Total number of trades (T_t).

    Returns:
        float: The win ratio. Returns 0.0 if total_trades is zero.
    """
    if total_trades == 0:
        # Avoid division by zero. If there are no trades, the win ratio is 0.
        # Also handles the case where winning_trades > 0 but total_trades = 0 (which shouldn't happen logically).
        return 0.0
    return winning_trades / total_trades

if __name__ == '__main__':
    print("--- Testing Evaluation Metrics ---")

    print("\nTesting calculate_daily_net_profit_margin:")
    # Test cases for calculate_daily_net_profit_margin
    p_margin_1 = calculate_daily_net_profit_margin(100.0, 110.0)
    print(f"Profit Margin (100 -> 110): {p_margin_1}") # Expect 0.1
    assert abs(p_margin_1 - 0.1) < 1e-9, f"Test failed: Expected 0.1, got {p_margin_1}"

    p_margin_2 = calculate_daily_net_profit_margin(100.0, 90.0)
    print(f"Profit Margin (100 -> 90): {p_margin_2}") # Expect -0.1
    assert abs(p_margin_2 - (-0.1)) < 1e-9, f"Test failed: Expected -0.1, got {p_margin_2}"

    p_margin_3 = calculate_daily_net_profit_margin(100.0, 100.0)
    print(f"Profit Margin (100 -> 100): {p_margin_3}") # Expect 0.0
    assert abs(p_margin_3 - 0.0) < 1e-9, f"Test failed: Expected 0.0, got {p_margin_3}"

    p_margin_4 = calculate_daily_net_profit_margin(0.0, 10.0)
    print(f"Profit Margin (0 -> 10): {p_margin_4}") # Expect 0.0
    assert abs(p_margin_4 - 0.0) < 1e-9, f"Test failed: Expected 0.0, got {p_margin_4}"
    
    p_margin_5 = calculate_daily_net_profit_margin(0.0, 0.0)
    print(f"Profit Margin (0 -> 0): {p_margin_5}") # Expect 0.0
    assert abs(p_margin_5 - 0.0) < 1e-9, f"Test failed: Expected 0.0, got {p_margin_5}"


    print("\nTesting calculate_win_ratio:")
    # Test cases for calculate_win_ratio
    win_ratio_1 = calculate_win_ratio(5, 10)
    print(f"Win Ratio (5 wins, 10 trades): {win_ratio_1}") # Expect 0.5
    assert abs(win_ratio_1 - 0.5) < 1e-9, f"Test failed: Expected 0.5, got {win_ratio_1}"

    win_ratio_2 = calculate_win_ratio(0, 10)
    print(f"Win Ratio (0 wins, 10 trades): {win_ratio_2}") # Expect 0.0
    assert abs(win_ratio_2 - 0.0) < 1e-9, f"Test failed: Expected 0.0, got {win_ratio_2}"

    win_ratio_3 = calculate_win_ratio(10, 10)
    print(f"Win Ratio (10 wins, 10 trades): {win_ratio_3}") # Expect 1.0
    assert abs(win_ratio_3 - 1.0) < 1e-9, f"Test failed: Expected 1.0, got {win_ratio_3}"

    win_ratio_4 = calculate_win_ratio(0, 0)
    print(f"Win Ratio (0 wins, 0 trades): {win_ratio_4}") # Expect 0.0
    assert abs(win_ratio_4 - 0.0) < 1e-9, f"Test failed: Expected 0.0, got {win_ratio_4}"

    win_ratio_5 = calculate_win_ratio(5, 0) # Logically inconsistent input but testing robustness
    print(f"Win Ratio (5 wins, 0 trades): {win_ratio_5}") # Expect 0.0
    assert abs(win_ratio_5 - 0.0) < 1e-9, f"Test failed: Expected 0.0, got {win_ratio_5}"
    
    print("\nAll evaluation metric tests passed.")
