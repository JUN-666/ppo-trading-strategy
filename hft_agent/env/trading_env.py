import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from ..utils.priority_queue import PricePriorityQueues # Relative import
import math
from datetime import date

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, 
                 data_df: pd.DataFrame, 
                 transaction_cost_config: dict, 
                 max_position: int = 10, 
                 pnl_magnification_C: float = 1.0,
                 initial_P_ask_past_default_offset: float = 0.0,
                 initial_P_bid_past_default_offset: float = 0.0,
                 action_reversal_config: dict = None,
                 initial_assets_ntd: float = 1_000_000.0,
                 is_evaluation: bool = False):
        super().__init__()

        self.data_df = data_df
        self.transaction_cost_config = transaction_cost_config
        self.max_position = max_position
        self.pnl_magnification_C = pnl_magnification_C
        self.initial_P_ask_past_default_offset = initial_P_ask_past_default_offset
        self.initial_P_bid_past_default_offset = initial_P_bid_past_default_offset
        self.action_reversal_config = action_reversal_config
        
        self.initial_assets_ntd = initial_assets_ntd
        self.is_evaluation = is_evaluation

        self.price_queues = PricePriorityQueues()
        
        self.current_step = 0
        self.current_position_qty = 0  # Q_t
        self.daily_pnl_normalized_reward = 0.0 # For training: sum of normalized rewards
        self.total_agent_actions = 0 # For training: count of all non-NO_ACTION agent actions

        # Evaluation specific tracking
        self.current_assets_ntd = self.initial_assets_ntd
        self.episode_total_round_trip_trades = 0 
        self.episode_winning_round_trip_trades = 0 
        self.last_entry_price_original = 0.0 

        self.action_reversal_active_today = False
        self.reversal_lower_bound = -np.inf
        self.reversal_upper_bound = np.inf
        self.current_P_open_for_reversal = None 
        self.current_P_dev_for_reversal = None 

        self.action_space = spaces.Discrete(3)
        low_obs = np.array([-np.inf] * 7 + [-1.0], dtype=np.float32)
        high_obs = np.array([np.inf] * 7 + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        if 'date' not in self.data_df.columns:
             # Attempt to create 'date' if 'timestamp' exists, ensuring it's datetime.date
             if 'timestamp' in self.data_df.columns and pd.api.types.is_datetime64_any_dtype(self.data_df['timestamp']):
                 self.data_df['date'] = self.data_df['timestamp'].dt.date
             else: # If no timestamp or not datetime, this will be an issue
                  raise ValueError("data_df must have a 'date' column of datetime.date objects, or a 'timestamp' column to derive it.")
        
        # Ensure self.data_df['date'] are datetime.date objects
        if not self.data_df.empty and not isinstance(self.data_df['date'].iloc[0], date):
            try:
                self.data_df['date'] = pd.to_datetime(self.data_df['date']).dt.date
            except Exception as e:
                raise ValueError(f"Failed to convert 'date' column to datetime.date objects: {e}")

        self.trading_days = sorted(self.data_df['date'].unique())
        if not self.trading_days:
            raise ValueError("No unique trading days found in data_df. Ensure 'date' column is present and populated.")
        self.current_day_index = -1 
        self.daily_data = pd.DataFrame()

    def _select_next_day(self, target_day_date=None):
        if target_day_date is not None:
            if not isinstance(target_day_date, date): # Ensure target_day_date is datetime.date
                target_day_date = pd.to_datetime(target_day_date).date()
            
            if target_day_date not in self.trading_days:
                available_days_str = ", ".join([d.isoformat() for d in self.trading_days[:5]]) + "..."
                raise ValueError(f"Target day {target_day_date.isoformat()} not found in available trading days. Available (sample): {available_days_str}")
            self.current_day_index = self.trading_days.index(target_day_date)
        else: 
            self.current_day_index = (self.current_day_index + 1) % len(self.trading_days)
        
        current_day_date_to_load = self.trading_days[self.current_day_index]
        self.daily_data = self.data_df[self.data_df['date'] == current_day_date_to_load].reset_index(drop=True)
        
        required_cols = ['norm_best_bid_price', 'norm_best_ask_price', 
                         'best_bid_price', 'best_ask_price', 'P_open', 'P_dev']
        if not all(col in self.daily_data.columns for col in required_cols):
            raise ValueError(f"Daily data for {current_day_date_to_load.isoformat()} is missing required columns.")
        if self.daily_data.empty:
            raise ValueError(f"No data found for day {current_day_date_to_load.isoformat()}.")

        self.current_P_open_for_reversal = self.daily_data['P_open'].iloc[0]
        self.current_P_dev_for_reversal = self.daily_data['P_dev'].iloc[0]

        if self.action_reversal_config and self.action_reversal_config.get('enable', False):
            self.action_reversal_active_today = True
            price_range_devs = self.action_reversal_config.get('price_range_devs', 0.5)
            self.reversal_lower_bound = self.current_P_open_for_reversal - (price_range_devs * self.current_P_dev_for_reversal)
            self.reversal_upper_bound = self.current_P_open_for_reversal + (price_range_devs * self.current_P_dev_for_reversal)
        else:
            self.action_reversal_active_today = False
            self.reversal_lower_bound = -np.inf
            self.reversal_upper_bound = np.inf

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed) 

        target_day_to_select = None
        if options and 'target_day' in options:
            target_day_to_select = options['target_day']
            # Ensure target_day_to_select is converted to datetime.date if it's a string or pd.Timestamp
            if isinstance(target_day_to_select, str):
                try:
                    target_day_to_select = pd.to_datetime(target_day_to_select).date()
                except ValueError:
                    raise ValueError(f"Invalid date string for target_day: {options['target_day']}")
            elif isinstance(target_day_to_select, pd.Timestamp):
                 target_day_to_select = target_day_to_select.date()
            elif not isinstance(target_day_to_select, date): # Check if it's not already a date object
                raise ValueError(f"target_day must be a date string, pd.Timestamp, or datetime.date object. Got: {type(target_day_to_select)}")


        self._select_next_day(target_day_date=target_day_to_select) 
        
        self.current_step = 0
        self.current_position_qty = 0
        self.price_queues.reset()
        self.daily_pnl_normalized_reward = 0.0 
        self.total_agent_actions = 0 
        
        self.current_assets_ntd = self.initial_assets_ntd
        self.episode_total_round_trip_trades = 0
        self.episode_winning_round_trip_trades = 0
        self.last_entry_price_original = 0.0 

        if self.daily_data.empty: # Should be caught by _select_next_day, but as a safeguard
            raise EnvironmentError("Failed to load data for the new episode in reset().")

        initial_observation = self._get_observation()
        info = self._get_info()
        
        return initial_observation, info

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.daily_data):
             empty_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
             return empty_obs

        current_tick_data = self.daily_data.iloc[self.current_step]
        P_bid_t_norm = current_tick_data['norm_best_bid_price']
        P_ask_t_norm = current_tick_data['norm_best_ask_price']
        spread_norm = P_ask_t_norm - P_bid_t_norm
        P_bid_past_state_norm = self.price_queues.get_current_P_bid_past()
        delta_P_buy_state = (P_bid_past_state_norm - P_ask_t_norm) if P_bid_past_state_norm is not None else self.initial_P_bid_past_default_offset 
        P_ask_past_state_norm = self.price_queues.get_current_P_ask_past()
        delta_P_sell_state = (P_bid_t_norm - P_ask_past_state_norm) if P_ask_past_state_norm is not None else -self.initial_P_ask_past_default_offset
        diff_bid_past_bid_t = (P_bid_past_state_norm - P_bid_t_norm) if P_bid_past_state_norm is not None else 0.0
        diff_ask_t_ask_past = (P_ask_t_norm - P_ask_past_state_norm) if P_ask_past_state_norm is not None else 0.0
        norm_position_qty = self.current_position_qty / self.max_position if self.max_position != 0 else 0.0

        obs = np.array([
            P_bid_t_norm, P_ask_t_norm, spread_norm, delta_P_buy_state, delta_P_sell_state,
            diff_bid_past_bid_t, diff_ask_t_ask_past, norm_position_qty
        ], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _calculate_transaction_cost(self, original_price_at_trade: float, return_ntd_cost: bool) -> float:
        # This check might be redundant if current_step is always valid when called
        if self.current_step >= len(self.daily_data): return 0.0 
        
        current_tick_data = self.daily_data.iloc[self.current_step]
        P_dev = current_tick_data['P_dev']
        contract_value = original_price_at_trade * 50 
        tax = contract_value * self.transaction_cost_config.get('tax_rate', 0.00002)
        handling_fee = self.transaction_cost_config.get('handling_fee', 7.5) 
        settlement_fee = self.transaction_cost_config.get('settlement_fee', 5.0)
        total_cost_ntd = handling_fee + settlement_fee + tax
        
        if return_ntd_cost:
            return total_cost_ntd
        else: # Return scaled cost for reward
            if P_dev == 0: return np.inf # Should not happen with valid P_dev
            return total_cost_ntd / P_dev

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_step >= len(self.daily_data) - 1: 
            obs = self._get_observation() 
            return obs, 0.0, True, False, self._get_info(is_done=True)

        current_tick_data = self.daily_data.iloc[self.current_step]
        info_extra = {'round_trip_trade_pnl_ntd': None} 
        
        if self.action_reversal_active_today:
            current_original_bid_price_val = current_tick_data['best_bid_price']
            current_original_ask_price_val = current_tick_data['best_ask_price']
            current_original_mid_price = (current_original_bid_price_val + current_original_ask_price_val) / 2.0
            if self.reversal_lower_bound <= current_original_mid_price <= self.reversal_upper_bound:
                if action == 1: action = 2 
                elif action == 2: action = 1
            else: self.action_reversal_active_today = False 
        
        P_bid_t_norm = current_tick_data['norm_best_bid_price'] 
        P_ask_t_norm = current_tick_data['norm_best_ask_price']
        original_P_bid_t = current_tick_data['best_bid_price'] 
        original_P_ask_t = current_tick_data['best_ask_price']

        reward = 0.0
        delta_P_buy_reward_component = 0.0  
        delta_P_sell_reward_component = 0.0 
        actual_tx_cost_this_step_scaled = 0.0
        
        previous_position_qty = self.current_position_qty

        if action == 1: # Buy Action
            # Allow buy if not at max long, OR if currently short (to reduce/close short)
            if self.current_position_qty < self.max_position :
                trade_price_original = original_P_ask_t 
                actual_tx_cost_this_step_scaled = self._calculate_transaction_cost(trade_price_original, return_ntd_cost=False)
                cost_ntd = self._calculate_transaction_cost(trade_price_original, return_ntd_cost=True)
                self.current_assets_ntd -= cost_ntd
                
                if previous_position_qty < 0: # Closing a short position
                    P_bid_past_norm = self.price_queues.get_bid_past_for_buy_close() 
                    delta_P_buy_reward_component = P_bid_past_norm - P_ask_t_norm if P_bid_past_norm is not None else 0.0
                    round_trip_profit_ntd = (self.last_entry_price_original - trade_price_original) * 50
                    self.current_assets_ntd += round_trip_profit_ntd 
                    info_extra['round_trip_trade_pnl_ntd'] = round_trip_profit_ntd - cost_ntd # PnL of this trade including its own cost
                    self.episode_total_round_trip_trades += 1
                    if (round_trip_profit_ntd - cost_ntd) > 0: self.episode_winning_round_trip_trades +=1
                else: # Opening or increasing a long position
                    self.price_queues.add_buy_trade(P_ask_t_norm) 
                    self.last_entry_price_original = trade_price_original 
                
                self.current_position_qty += 1
                self.total_agent_actions += 1 

        elif action == 2: # Sell Action
             # Allow sell if not at max short, OR if currently long (to reduce/close long)
             if self.current_position_qty > -self.max_position:
                trade_price_original = original_P_bid_t 
                actual_tx_cost_this_step_scaled = self._calculate_transaction_cost(trade_price_original, return_ntd_cost=False)
                cost_ntd = self._calculate_transaction_cost(trade_price_original, return_ntd_cost=True)
                self.current_assets_ntd -= cost_ntd

                if previous_position_qty > 0: # Closing a long position
                    P_ask_past_norm = self.price_queues.get_ask_past_for_sell_close() 
                    delta_P_sell_reward_component = P_bid_t_norm - P_ask_past_norm if P_ask_past_norm is not None else 0.0
                    round_trip_profit_ntd = (trade_price_original - self.last_entry_price_original) * 50
                    self.current_assets_ntd += round_trip_profit_ntd
                    info_extra['round_trip_trade_pnl_ntd'] = round_trip_profit_ntd - cost_ntd
                    self.episode_total_round_trip_trades += 1
                    if (round_trip_profit_ntd - cost_ntd) > 0: self.episode_winning_round_trip_trades +=1
                else: # Opening or increasing a short position
                    self.price_queues.add_sell_trade(P_bid_t_norm)
                    self.last_entry_price_original = trade_price_original
                
                self.current_position_qty -= 1
                self.total_agent_actions += 1
        
        if not self.is_evaluation: 
            denominator_log_term = 2 + abs(self.current_position_qty)
            if denominator_log_term <= 1: denominator_log_term = 1.01 
            A_magnifier = self.pnl_magnification_C / math.log2(denominator_log_term)
            if action == 1 and (previous_position_qty < self.max_position) : # If a buy trade actually happened
                 reward = A_magnifier * delta_P_buy_reward_component - actual_tx_cost_this_step_scaled
            elif action == 2 and (previous_position_qty > -self.max_position): # If a sell trade actually happened
                 reward = A_magnifier * delta_P_sell_reward_component - actual_tx_cost_this_step_scaled
            self.daily_pnl_normalized_reward += reward

        self.current_step += 1
        done = self.current_step >= len(self.daily_data) -1 
        next_observation = self._get_observation()
        info = self._get_info(is_done=done, round_trip_pnl_ntd=info_extra['round_trip_trade_pnl_ntd'])
        truncated = False 

        return next_observation, reward, done, truncated, info

    def _get_info(self, is_done=False, round_trip_pnl_ntd=None) -> dict:
        info = {
            "current_step": self.current_step,
            "current_position_qty": self.current_position_qty,
            "daily_pnl_normalized_reward": self.daily_pnl_normalized_reward, 
            "total_agent_actions": self.total_agent_actions,
            "buy_queue_size": self.price_queues.get_buy_queue_size(),
            "sell_queue_size": self.price_queues.get_sell_queue_size(),
            "action_reversal_active": self.action_reversal_active_today,
            "current_assets_ntd": self.current_assets_ntd, 
            "episode_total_round_trip_trades": self.episode_total_round_trip_trades,
            "episode_winning_round_trip_trades": self.episode_winning_round_trip_trades
        }
        if round_trip_pnl_ntd is not None:
            info['round_trip_trade_pnl_ntd'] = round_trip_pnl_ntd
        if is_done:
            info["episode_end_reason"] = "end_of_day"
            info["final_assets_ntd"] = self.current_assets_ntd 
        return info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}, Position: {self.current_position_qty}, Assets (NTD): {self.current_assets_ntd:.2f}")
        else:
            super().render(mode=mode)

    def close(self):
        print("TradingEnv closed.")

if __name__ == '__main__':
    print("Setting up a dummy TradingEnv for testing PnL and evaluation...")
    num_ticks_day1 = 200
    num_ticks_day2 = 150
    data_day1 = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 09:30:00'] * num_ticks_day1) + pd.to_timedelta(np.arange(num_ticks_day1), unit='s'),
        'norm_best_bid_price': np.linspace(0.1, 0.15, num_ticks_day1), 'norm_best_ask_price': np.linspace(0.105, 0.155, num_ticks_day1),
        'best_bid_price': np.linspace(10000, 10005, num_ticks_day1), 'best_ask_price': np.linspace(10000.5, 10005.5, num_ticks_day1),
        'P_open': [10000.0] * num_ticks_day1, 'P_dev': [64.0] * num_ticks_day1, 'date': pd.to_datetime('2023-01-01').date()
    })
    data_day1['norm_best_ask_price'] = np.maximum(data_day1['norm_best_ask_price'], data_day1['norm_best_bid_price'] + 0.0001)
    data_day2 = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-02 09:30:00'] * num_ticks_day2) + pd.to_timedelta(np.arange(num_ticks_day2), unit='s'),
        'norm_best_bid_price': np.linspace(0.12, 0.17, num_ticks_day2), 
        'norm_best_ask_price': np.linspace(0.125, 0.175, num_ticks_day2),
        'best_bid_price': np.linspace(10050, 10055, num_ticks_day2), 'best_ask_price': np.linspace(10050.5, 10055.5, num_ticks_day2),
        'P_open': [10050.0] * num_ticks_day2, 'P_dev': [64.0] * num_ticks_day2, 'date': pd.to_datetime('2023-01-02').date()
    })
    data_day2['norm_best_ask_price'] = np.maximum(data_day2['norm_best_ask_price'], data_day2['norm_best_bid_price'] + 0.0001)
    sample_df = pd.concat([data_day1, data_day2], ignore_index=True)
    transaction_costs = {'handling_fee': 7.5, 'settlement_fee': 5.0, 'tax_rate': 0.00002}
    
    env = TradingEnv(data_df=sample_df.copy(), transaction_cost_config=transaction_costs, initial_assets_ntd=1_000_000.0, is_evaluation=True)

    # Test reset and specific day selection
    target_date_obj = pd.to_datetime('2023-01-02').date()
    obs, info = env.reset(options={'target_day': target_date_obj})
    print(f"Reset to day: {env.daily_data['date'].iloc[0]}, Initial Assets: {info['current_assets_ntd']}")
    assert env.daily_data['date'].iloc[0] == target_date_obj
    assert info['current_assets_ntd'] == 1_000_000.0

    print("\nTesting Buy (Open Long) -> Sell (Close Long) sequence...")
    action_buy = 1
    obs, reward_buy, _, _, info_buy = env.step(action_buy)
    entry_price_original = env.last_entry_price_original
    cost_buy_ntd = env.initial_assets_ntd - info_buy['current_assets_ntd']
    print(f"After Buy: Pos={info_buy['current_position_qty']}, Assets={info_buy['current_assets_ntd']:.2f}, EntryPrice={entry_price_original:.2f}, Cost={cost_buy_ntd:.2f}")
    assert info_buy['current_position_qty'] == 1
    
    action_sell = 2
    # Manually get the sell price from the *next* tick for accurate cost calculation if step() uses current_tick_data for trade execution
    # If step() has already advanced current_step for the *next* observation but used *previous* current_step's data for trade, this is tricky.
    # The current env.step() uses data at self.current_step to execute the trade, then increments self.current_step.
    # So, the original_P_bid_t for the sell trade was from current_tick_data at the time of the sell.
    # Need to capture it before the info_sell from env.step(action_sell) is obtained, or use the info from the step if it provides trade price.
    # For simplicity, the test currently uses daily_data.iloc[env.current_step-1] which is correct as current_step was incremented.
    
    # Simulate one more step to make sure we are not at the end of data for the sell
    if env.current_step < len(env.daily_data) -1:
        obs, reward_sell, _, _, info_sell = env.step(action_sell)
        exit_price_original = env.daily_data.iloc[env.current_step-1]['best_bid_price'] 
        
        # cost_sell_ntd is the cost of the sell trade itself.
        # current_assets_ntd before this sell trade was info_buy['current_assets_ntd']
        # current_assets_ntd after this sell trade is info_sell['current_assets_ntd']
        # The change in assets due to this sell operation is:
        # (profit_from_price_diff) - cost_of_sell_trade
        # So, info_sell['current_assets_ntd'] = info_buy['current_assets_ntd'] + (profit_from_price_diff) - cost_of_sell_trade
        # cost_of_sell_trade = info_buy['current_assets_ntd'] + profit_from_price_diff - info_sell['current_assets_ntd']
        
        expected_pnl_price_diff = (exit_price_original - entry_price_original) * 50
        # The cost of this sell trade is what _calculate_transaction_cost would return for exit_price_original
        cost_sell_ntd = env._calculate_transaction_cost(exit_price_original, return_ntd_cost=True)

        reported_round_trip_pnl = info_sell['round_trip_trade_pnl_ntd']
        
        print(f"After Sell: Pos={info_sell['current_position_qty']}, Assets={info_sell['current_assets_ntd']:.2f}, ExitPrice={exit_price_original:.2f}, Cost of Sell Trade={cost_sell_ntd:.2f}")
        print(f"  Expected PnL from price diff (gross): {expected_pnl_price_diff:.2f}")
        print(f"  Reported round_trip_trade_pnl_ntd (net of its own cost): {reported_round_trip_pnl:.2f}")
        
        assert info_sell['current_position_qty'] == 0
        assert info_sell['episode_total_round_trip_trades'] == 1
        
        expected_final_assets = env.initial_assets_ntd - cost_buy_ntd + expected_pnl_price_diff - cost_sell_ntd
        assert abs(info_sell['current_assets_ntd'] - expected_final_assets) < 1e-6, f"Asset mismatch: expected {expected_final_assets:.2f}, got {info_sell['current_assets_ntd']:.2f}"
        assert abs(reported_round_trip_pnl - (expected_pnl_price_diff - cost_sell_ntd)) < 1e-6, "Round trip PnL mismatch"
        
        if (expected_pnl_price_diff - cost_sell_ntd) > 0 :
            assert info_sell['episode_winning_round_trip_trades'] == 1
        else:
            assert info_sell['episode_winning_round_trip_trades'] == 0
        print("PnL tracking test for Long sequence passed.")
    else:
        print("Skipping sell part of PnL test as not enough data points for two steps.")
        
    env.close()
    print("\nTradingEnv evaluation features tests completed.")
