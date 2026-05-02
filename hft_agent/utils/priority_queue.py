import heapq

class PricePriorityQueues:
    """
    Manages two priority queues for past trade prices:
    1. buy_prices_pq: Stores ask prices of past BUY trades (min-heap).
       Used to find the P_Ask,past for closing a long position (selling).
    2. sell_prices_pq: Stores bid prices of past SELL trades (max-heap, implemented with negative values in a min-heap).
       Used to find the P_Bid,past for closing a short position (buying back).
    """

    def __init__(self):
        """
        Initializes two empty lists for the priority queues.
        """
        self.buy_prices_pq = []  # Min-heap for ask prices of BUY trades
        self.sell_prices_pq = [] # Max-heap for bid prices of SELL trades (stores -price)

    def add_buy_trade(self, ask_price: float):
        """
        Adds the ask price of a BUY trade to the buy_prices_pq.
        Args:
            ask_price: The ask price at which the buy trade was executed.
        """
        heapq.heappush(self.buy_prices_pq, ask_price)

    def add_sell_trade(self, bid_price: float):
        """
        Adds the bid price of a SELL trade to the sell_prices_pq.
        Stores as negative to simulate a max-heap with Python's min-heap.
        Args:
            bid_price: The bid price at which the sell trade was executed.
        """
        heapq.heappush(self.sell_prices_pq, -bid_price)

    def get_ask_past_for_sell_close(self) -> float | None:
        """
        Retrieves and removes the smallest ask price from past BUY trades.
        This price (P_Ask,past) is used when closing a long position (selling).
        Returns:
            The smallest ask price, or None if no past buy trades exist.
        """
        if not self.buy_prices_pq:
            return None
        return heapq.heappop(self.buy_prices_pq)

    def get_bid_past_for_buy_close(self) -> float | None:
        """
        Retrieves and removes the largest bid price from past SELL trades.
        This price (P_Bid,past) is used when closing a short position (buying back).
        Returns:
            The largest bid price, or None if no past sell trades exist.
        """
        if not self.sell_prices_pq:
            return None
        return -heapq.heappop(self.sell_prices_pq) # Negate back to get original price

    def get_current_P_ask_past(self) -> float | None:
        """
        Peeks at the smallest ask price from past BUY trades without removing it.
        Used for state representation.
        Returns:
            The smallest ask price, or None if no past buy trades exist.
        """
        if not self.buy_prices_pq:
            return None
        return self.buy_prices_pq[0] # Smallest element in a min-heap is at index 0

    def get_current_P_bid_past(self) -> float | None:
        """
        Peeks at the largest bid price from past SELL trades without removing it.
        Used for state representation.
        Returns:
            The largest bid price, or None if no past sell trades exist.
        """
        if not self.sell_prices_pq:
            return None
        return -self.sell_prices_pq[0] # Smallest negative is largest positive

    def get_buy_queue_size(self) -> int:
        """
        Returns the number of past BUY trades stored.
        """
        return len(self.buy_prices_pq)

    def get_sell_queue_size(self) -> int:
        """
        Returns the number of past SELL trades stored.
        """
        return len(self.sell_prices_pq)

    def reset(self):
        """
        Clears all stored trade prices from both queues.
        """
        self.buy_prices_pq = []
        self.sell_prices_pq = []

if __name__ == '__main__':
    print("Testing PricePriorityQueues...")
    queues = PricePriorityQueues()

    # Test initial state
    print(f"\nInitial buy queue size: {queues.get_buy_queue_size()}") # Expected: 0
    assert queues.get_buy_queue_size() == 0
    print(f"Initial sell queue size: {queues.get_sell_queue_size()}") # Expected: 0
    assert queues.get_sell_queue_size() == 0
    print(f"Initial P_ask_past (peek): {queues.get_current_P_ask_past()}") # Expected: None
    assert queues.get_current_P_ask_past() is None
    print(f"Initial P_bid_past (peek): {queues.get_current_P_bid_past()}") # Expected: None
    assert queues.get_current_P_bid_past() is None
    print(f"Initial ask_past for sell close (pop): {queues.get_ask_past_for_sell_close()}") # Expected: None
    assert queues.get_ask_past_for_sell_close() is None
    print(f"Initial bid_past for buy close (pop): {queues.get_bid_past_for_buy_close()}") # Expected: None
    assert queues.get_bid_past_for_buy_close() is None

    # Test adding buy trades
    print("\nAdding BUY trades...")
    queues.add_buy_trade(100.5)
    queues.add_buy_trade(100.2) # This should be the smallest
    queues.add_buy_trade(100.8)
    print(f"Buy queue size after adds: {queues.get_buy_queue_size()}") # Expected: 3
    assert queues.get_buy_queue_size() == 3
    print(f"Current P_ask_past (peek): {queues.get_current_P_ask_past()}") # Expected: 100.2
    assert queues.get_current_P_ask_past() == 100.2

    # Test adding sell trades
    print("\nAdding SELL trades...")
    queues.add_sell_trade(99.5)
    queues.add_sell_trade(99.8) # This should be the largest
    queues.add_sell_trade(99.2)
    print(f"Sell queue size after adds: {queues.get_sell_queue_size()}") # Expected: 3
    assert queues.get_sell_queue_size() == 3
    print(f"Current P_bid_past (peek): {queues.get_current_P_bid_past()}") # Expected: 99.8
    assert queues.get_current_P_bid_past() == 99.8

    # Test closing positions (popping)
    print("\nClosing positions (popping)...")
    # Close a long position (sell) - should get smallest ask_price from buy_prices_pq
    ask_to_close_long = queues.get_ask_past_for_sell_close()
    print(f"Popped ask_past for sell close: {ask_to_close_long}") # Expected: 100.2
    assert ask_to_close_long == 100.2
    print(f"Buy queue size after pop: {queues.get_buy_queue_size()}") # Expected: 2
    assert queues.get_buy_queue_size() == 2
    print(f"Current P_ask_past (peek) after pop: {queues.get_current_P_ask_past()}") # Expected: 100.5
    assert queues.get_current_P_ask_past() == 100.5

    # Close a short position (buy back) - should get largest bid_price from sell_prices_pq
    bid_to_close_short = queues.get_bid_past_for_buy_close()
    print(f"Popped bid_past for buy close: {bid_to_close_short}") # Expected: 99.8
    assert bid_to_close_short == 99.8
    print(f"Sell queue size after pop: {queues.get_sell_queue_size()}") # Expected: 2
    assert queues.get_sell_queue_size() == 2
    print(f"Current P_bid_past (peek) after pop: {queues.get_current_P_bid_past()}") # Expected: 99.5
    assert queues.get_current_P_bid_past() == 99.5

    # Pop remaining items
    print("\nPopping remaining items...")
    print(f"Popped ask_past: {queues.get_ask_past_for_sell_close()}") # Expected: 100.5
    assert queues.get_current_P_ask_past() == 100.8 # Check before it's gone
    assert queues.get_ask_past_for_sell_close() == 100.8 # Expected: 100.8
    assert queues.get_ask_past_for_sell_close() is None # Expected: None (empty)
    assert queues.get_buy_queue_size() == 0

    print(f"Popped bid_past: {queues.get_bid_past_for_buy_close()}") # Expected: 99.5
    assert queues.get_current_P_bid_past() == 99.2 # Check before it's gone
    assert queues.get_bid_past_for_buy_close() == 99.2 # Expected: 99.2
    assert queues.get_bid_past_for_buy_close() is None # Expected: None (empty)
    assert queues.get_sell_queue_size() == 0

    # Verify queues are empty
    print("\nVerifying queues are empty...")
    print(f"Buy queue size: {queues.get_buy_queue_size()}") # Expected: 0
    assert queues.get_buy_queue_size() == 0
    print(f"Sell queue size: {queues.get_sell_queue_size()}") # Expected: 0
    assert queues.get_sell_queue_size() == 0
    print(f"P_ask_past (peek): {queues.get_current_P_ask_past()}") # Expected: None
    assert queues.get_current_P_ask_past() is None
    print(f"P_bid_past (peek): {queues.get_current_P_bid_past()}") # Expected: None
    assert queues.get_current_P_bid_past() is None

    # Test reset functionality
    print("\nTesting reset...")
    queues.add_buy_trade(101.0)
    queues.add_sell_trade(98.0)
    print(f"Buy queue size before reset: {queues.get_buy_queue_size()}") # Expected: 1
    assert queues.get_buy_queue_size() == 1
    print(f"Sell queue size before reset: {queues.get_sell_queue_size()}") # Expected: 1
    assert queues.get_sell_queue_size() == 1
    
    queues.reset()
    print("Queues reset.")
    print(f"Buy queue size after reset: {queues.get_buy_queue_size()}") # Expected: 0
    assert queues.get_buy_queue_size() == 0
    print(f"Sell queue size after reset: {queues.get_sell_queue_size()}") # Expected: 0
    assert queues.get_sell_queue_size() == 0
    print(f"P_ask_past (peek) after reset: {queues.get_current_P_ask_past()}") # Expected: None
    assert queues.get_current_P_ask_past() is None
    print(f"P_bid_past (peek) after reset: {queues.get_current_P_bid_past()}") # Expected: None
    assert queues.get_current_P_bid_past() is None

    print("\nAll tests for PricePriorityQueues passed.")
