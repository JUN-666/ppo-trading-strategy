import torch
import numpy as np
import pandas as pd
from hft_agent.env.trading_env import TradingEnv
from hft_agent.ppo.ppo_agent import PPOAgent
from hft_agent.utils.data_loader import load_tick_data, normalize_prices
import os 

def train_ppo_agent():
    # 1. Configuration
    data_file = None 
    log_interval = 1  # Reduced for quick logging
    max_episodes = 2    # Drastically reduced for quick execution test
    max_timesteps_per_episode = 50 # Reduced timesteps for faster episodes
    
    # PPO Hyperparameters
    lr = 0.0003
    gamma = 0.99
    ppo_epsilon = 0.2
    ppo_epochs = 1 # Reduced PPO epochs for faster updates
    # PPO agent updates after its internal memory (buffer) collects this many steps.
    ppo_agent_internal_batch_size = 50 # Reduced for more frequent updates in short run
    c1_value_loss_coeff = 0.5 
    c2_entropy_coeff = 0.01  
    
    # Environment/Agent Config
    initial_assets_ntd = 1_000_000 
    transaction_cost_config = {'handling_fee': 7.5, 'settlement_fee': 5.0, 'tax_rate': 0.00002}
    max_position = 5
    pnl_magnification_C = 1.0 # For reward scaling in training
    action_reversal_config = {'enable': True, 'price_range_devs': 0.2} 

    model_save_dir = "./trained_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, "ppo_actor_critic_hft.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load and Preprocess Data
    print("Loading and preprocessing data...")
    # load_tick_data will simulate data if data_file is None.
    # It creates 'timestamp' and 'date' (datetime.date objects) columns.
    raw_data = load_tick_data(file_path_or_config=data_file) 
    
    # normalize_prices uses 'P_open' (first mid-price at/after 09:30) and 'P_dev'.
    # It adds 'norm_best_bid_price', 'norm_best_ask_price', 'P_open', 'P_dev' columns.
    # It expects 'date' column to group by day for P_open calculation.
    normalized_data = normalize_prices(raw_data.copy(), price_dev=64.0)
    
    # 3. Initialize Environment and Agent
    env = TradingEnv(
        data_df=normalized_data, 
        transaction_cost_config=transaction_cost_config,
        initial_assets_ntd=initial_assets_ntd,
        max_position=max_position,
        pnl_magnification_C=pnl_magnification_C,
        action_reversal_config=action_reversal_config,
        is_evaluation=False # This is a training run
    )
    
    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.n
    
    agent = PPOAgent(
        state_dims=state_dims,
        action_dims=action_dims,
        lr=lr,
        gamma=gamma,
        ppo_epsilon=ppo_epsilon,
        ppo_epochs=ppo_epochs,
        batch_size=ppo_agent_internal_batch_size, # PPOAgent's internal batch size for update trigger
        c1_value_loss_coeff=c1_value_loss_coeff,
        c2_entropy_coeff=c2_entropy_coeff,
        device=device
    )

    # 4. Training Loop
    print("Starting training...")
    running_reward_log = 0 
    total_steps_overall = 0

    for episode in range(1, max_episodes + 1):
        state, info = env.reset() # Env selects a day, either next or random (if not eval)
        current_episode_reward = 0
        current_episode_steps = 0

        for t in range(1, max_timesteps_per_episode + 1):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            agent.store_transition(state, action, log_prob, reward, next_state, done or truncated)
            state = next_state
            current_episode_reward += reward
            total_steps_overall += 1
            current_episode_steps +=1

            # PPO Update is triggered when agent's memory meets its batch_size
            if len(agent.memory) >= agent.batch_size:
                # print(f"Episode {episode}, Step {t}: Agent memory full ({len(agent.memory)} / {agent.batch_size}), triggering PPO update.")
                agent.update() # agent.update() also clears its memory
            
            if done or truncated:
                break
        
        running_reward_log += current_episode_reward

        if episode % log_interval == 0:
            avg_reward_interval = running_reward_log / log_interval
            print(f"Episode {episode} | Avg Reward (last {log_interval} eps): {avg_reward_interval:.2f} | Last Ep Reward: {current_episode_reward:.2f} | Steps: {current_episode_steps}")
            running_reward_log = 0
        
        # Periodically save model
        if episode % (log_interval * 10) == 0: # e.g., every 100 episodes for log_interval=10
            print(f"Saving model at episode {episode}")
            agent.save_model(model_save_path)

    env.close()
    print("Training finished.")
    agent.save_model(model_save_path) 
    print(f"Final model saved to {model_save_path}")

if __name__ == '__main__':
    train_ppo_agent()
