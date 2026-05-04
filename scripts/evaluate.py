import torch
import numpy as np
import pandas as pd
from hft_agent.env.trading_env import TradingEnv
from hft_agent.ppo.ppo_agent import PPOAgent 
from hft_agent.utils.data_loader import load_tick_data, normalize_prices
from hft_agent.utils.evaluation import calculate_daily_net_profit_margin, calculate_win_ratio
from hft_agent.quantization.ptq import convert_model_to_quantized_simulation 
import os
from datetime import date # For type hinting if needed

def evaluate_ppo_agent():
    # 1. Configuration
    data_file = None 
    model_load_path = "./trained_models/ppo_actor_critic_hft.pth" 

    # Quantization options
    quantize_enabled = False # Set to True to evaluate quantized model
    quantization_configs = { 
        'result1': { 
            'fc1': {'weight_bits': 8, 'activation_bits': 8, 'bias_bits': 8},
            'fc2': {'weight_bits': 5, 'activation_bits': 5, 'bias_bits': 5},
            'actor_head': {'weight_bits': 8, 'activation_bits': 8, 'bias_bits': 8},
            'critic_head': {'weight_bits': 8, 'activation_bits': 8, 'bias_bits': 8}
        },
        'result2': { 
            'fc1': {'weight_bits': 8, 'activation_bits': 8, 'bias_bits': 8},
            'fc2': {'weight_bits': 3, 'activation_bits': 3, 'bias_bits': 3},
            'actor_head': {'weight_bits': 8, 'activation_bits': 8, 'bias_bits': 8},
            'critic_head': {'weight_bits': 8, 'activation_bits': 8, 'bias_bits': 8}
        }
    }
    selected_quant_config_key = 'result1' 

    initial_assets_ntd = 1_000_000.0
    transaction_cost_config = {'handling_fee': 7.5, 'settlement_fee': 5.0, 'tax_rate': 0.00002}
    max_position = 5
    pnl_magnification_C = 1.0 # Not directly used for eval metrics but part of env setup
    action_reversal_config = {'enable': True, 'price_range_devs': 0.2} 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading and preprocessing evaluation data...")
    raw_data = load_tick_data(file_path_or_config=data_file) 
    normalized_data = normalize_prices(raw_data.copy(), price_dev=64.0)
    
    # Ensure 'date' column for TradingEnv daily iteration (load_tick_data should handle this)
    if 'date' not in normalized_data.columns:
        if 'timestamp' in normalized_data.columns:
            normalized_data['date'] = normalized_data['timestamp'].dt.date
        else: # Should not happen if load_tick_data is correct
            raise ValueError("Dataframe must contain a 'date' or 'timestamp' column.")
        
    # 3. Initialize Environment
    env = TradingEnv(
        data_df=normalized_data,
        transaction_cost_config=transaction_cost_config,
        initial_assets_ntd=initial_assets_ntd,
        max_position=max_position,
        pnl_magnification_C=pnl_magnification_C, 
        action_reversal_config=action_reversal_config,
        is_evaluation=True 
    )

    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.n

    # 4. Load Agent
    print(f"Loading model from {model_load_path}...")
    if not os.path.exists(model_load_path):
        print(f"Error: Model not found at {model_load_path}. Train a model first or check path.")
        return

    agent_to_evaluate = PPOAgent(state_dims, action_dims, device=device, lr=0.0003) 
    agent_to_evaluate.load_model(model_load_path) 
    agent_to_evaluate.actor_critic.eval() 

    # 5. Optionally Quantize Model
    if quantize_enabled:
        print(f"Applying PTQ simulation with config: {selected_quant_config_key}")
        if selected_quant_config_key not in quantization_configs:
            print(f"Error: Quantization config '{selected_quant_config_key}' not found.")
            return
        selected_config = quantization_configs[selected_quant_config_key]
        
        quantized_actor_critic = convert_model_to_quantized_simulation(
            agent_to_evaluate.actor_critic, 
            selected_config
        )
        agent_to_evaluate.actor_critic = quantized_actor_critic.to(device)
        agent_to_evaluate.actor_critic.eval() 
        print("Model converted to PTQ simulation mode.")

    # 6. Evaluation Loop
    print("Starting evaluation...")
    all_day_profit_margins = []
    all_day_win_ratios = []
    all_day_total_round_trip_trades = [] # Corrected from all_day_total_trades
    
    unique_days_in_data = sorted(normalized_data['date'].unique())
    if not unique_days_in_data:
        print("No unique days found in the evaluation data.")
        return

    for day_to_eval in unique_days_in_data:
        print(f"Evaluating for day: {day_to_eval}")
        state, info = env.reset(options={'target_day': day_to_eval}) 
        
        # Initial assets for the day is env.initial_assets_ntd (set by env.reset via its own init)
        # Final assets will be info.get('final_assets_ntd') at the end of the episode

        while True:
            action, _ = agent_to_evaluate.select_action(state) 
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            
            if done or truncated:
                break
        
        # Metrics are now directly from final info after episode ends
        daily_final_assets = info.get('final_assets_ntd', env.initial_assets_ntd) 
        day_profit_margin = calculate_daily_net_profit_margin(env.initial_assets_ntd, daily_final_assets)
        
        day_total_round_trips = info.get('episode_total_round_trip_trades', 0)
        day_winning_round_trips = info.get('episode_winning_round_trip_trades', 0)
        day_win_ratio = calculate_win_ratio(day_winning_round_trips, day_total_round_trips)
        
        print(f"Day {day_to_eval} - Final Assets: {daily_final_assets:.2f} NTD | Profit Margin: {day_profit_margin:.4f} | Win Ratio: {day_win_ratio:.2f} ({day_winning_round_trips}/{day_total_round_trips} round trips)")
        
        all_day_profit_margins.append(day_profit_margin)
        if day_total_round_trips > 0: 
            all_day_win_ratios.append(day_win_ratio)
        all_day_total_round_trip_trades.append(day_total_round_trips)

    env.close()
    
    num_eval_days = len(all_day_profit_margins)
    if num_eval_days > 0:
        avg_net_profit_margin = np.mean(all_day_profit_margins)
        avg_win_ratio = np.mean(all_day_win_ratios) if all_day_win_ratios else 0.0 
        avg_round_trip_trades_per_day = np.mean(all_day_total_round_trip_trades)

        print(f"\n--- Evaluation Summary ({num_eval_days} days) ---")
        print(f"Average Daily Net Profit Margin: {avg_net_profit_margin:.4f}")
        print(f"Average Daily Win Ratio (of days with trades): {avg_win_ratio:.2f}")
        print(f"Average Round Trip Trades per Day: {avg_round_trip_trades_per_day:.2f}")
    else:
        print("No evaluation days processed or no trades made to calculate aggregate metrics.")

if __name__ == '__main__':
    evaluate_ppo_agent()
