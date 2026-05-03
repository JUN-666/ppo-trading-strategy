import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
# Adjust the import path based on the actual project structure
# Assuming ppo_agent.py is in hft_agent/ppo/ and actor_critic.py is in hft_agent/models/
from ..models.actor_critic import ActorCritic

class PPOAgent:
    def __init__(self, 
                 state_dims: int, 
                 action_dims: int, 
                 lr: float = 3e-4, 
                 gamma: float = 0.99, 
                 ppo_epsilon: float = 0.2, 
                 ppo_epochs: int = 10, 
                 batch_size: int = 64, 
                 c1_value_loss_coeff: float = 0.5, 
                 c2_entropy_coeff: float = 0.01, 
                 device: str = 'cpu'):
        """
        Initializes the PPO Agent.

        Args:
            state_dims (int): Dimensionality of the state space.
            action_dims (int): Dimensionality of the action space.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            ppo_epsilon (float): Clipping parameter for PPO.
            ppo_epochs (int): Number of epochs to update the policy with the same batch of data.
            batch_size (int): Batch size for training.
            c1_value_loss_coeff (float): Coefficient for the value loss term.
            c2_entropy_coeff (float): Coefficient for the entropy bonus term.
            device (str): Device to run the computations on ('cpu' or 'cuda').
        """
        self.gamma = gamma
        self.ppo_epsilon = ppo_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size # Minimum batch size to trigger update, actual processing uses all memory
        self.c1_value_loss_coeff = c1_value_loss_coeff
        self.c2_entropy_coeff = c2_entropy_coeff
        
        self.device = torch.device(device)
        
        self.actor_critic = ActorCritic(input_dims=state_dims, actor_output_dims=action_dims).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.memory = [] # Stores (state, action, log_prob, reward, next_state, done)

    def store_transition(self, state, action, log_prob, reward, next_state, done):
        """Stores a single transition in the agent's memory."""
        self.memory.append((state, action, log_prob, reward, next_state, done))

    def clear_memory(self):
        """Clears the agent's memory."""
        self.memory = []

    def select_action(self, state: np.ndarray) -> tuple[int, float]:
        """
        Selects an action based on the current policy and state.

        Args:
            state (np.ndarray): The current state.

        Returns:
            tuple[int, float]: The selected action and its log probability.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor_critic.eval() # Set to evaluation mode
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_tensor)
        self.actor_critic.train() # Set back to training mode
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()

    def _calculate_discounted_returns(self, rewards: list[float], dones: list[bool], last_value_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates the discounted returns (G_t) for a batch of trajectories.
        G_t = sum_{k=t}^{N} gamma^(k-t) * r_k
        If an episode truncates (not done but data ends), last_value is bootstrapped.
        """
        returns = []
        # Bootstrap with the value of the last state if the episode didn't end there.
        # If dones[-1] is True, it means the episode terminated at the last step, so the future reward is 0.
        # Otherwise, last_value_tensor contains V(s_N) where N is the last step in memory.
        R = last_value_tensor.item() if not dones[-1] else 0.0
        
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done: # If it's a terminal state
                R = 0.0 # Reset discounted reward
            R = r + self.gamma * R # Standard discounted return calculation
            returns.insert(0, R)
            
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def update(self):
        """
        Updates the policy and value function using the PPO algorithm.
        """
        if len(self.memory) < self.batch_size: # Ensure enough data is collected
            return

        # Unpack memory
        states_list = [t[0] for t in self.memory]
        actions_list = [t[1] for t in self.memory]
        old_log_probs_list = [t[2] for t in self.memory]
        rewards_list = [t[3] for t in self.memory]
        next_states_list = [t[4] for t in self.memory] # Needed for last_value
        dones_list = [t[5] for t in self.memory]

        states = torch.FloatTensor(np.array(states_list)).to(self.device)
        actions = torch.LongTensor(actions_list).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs_list).to(self.device)
        # rewards and dones are kept as lists for _calculate_discounted_returns

        # Calculate Returns (G_t) and Advantages (A_t)
        # V(s_t) (current_values) and V(s_N) (last_value) are needed.
        # These are V_theta_old(s_t) - values from the policy used to collect data.
        self.actor_critic.eval() # Use the "old" policy for these calculations
        with torch.no_grad():
            # Get V(s_t) for all states in the batch
            _, current_values_detached = self.actor_critic(states)
            current_values_detached = current_values_detached.squeeze()

            # Get V(s_N) for the very last next_state in the memory batch
            # This is used to bootstrap returns if the last trajectory was truncated.
            if not dones_list[-1]: # If the last episode was not terminal
                last_next_state_tensor = torch.FloatTensor(next_states_list[-1]).unsqueeze(0).to(self.device)
                _, last_value_tensor_detached = self.actor_critic(last_next_state_tensor)
                last_value_tensor_detached = last_value_tensor_detached.squeeze(0) # remove batch dim
            else: # If the last episode was terminal, the value of the terminal state is 0
                last_value_tensor_detached = torch.tensor([0.0]).to(self.device)
        self.actor_critic.train() # Switch back to train mode for updates

        returns_g_t = self._calculate_discounted_returns(rewards_list, dones_list, last_value_tensor_detached)
        
        # Advantages: A_t = G_t - V(s_t) (where V(s_t) are from the old policy)
        advantages = returns_g_t - current_values_detached
        
        # Normalize advantages (optional but common)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update Loop
        for _ in range(self.ppo_epochs):
            # Get new action log probabilities and state values from the current (updating) policy
            action_probs_new, state_values_new = self.actor_critic(states)
            state_values_new = state_values_new.squeeze()
            
            dist_new = Categorical(action_probs_new)
            new_log_probs = dist_new.log_prob(actions)

            # Actor Loss (PPO-Clip Objective - Eq. 4)
            ratio = torch.exp(new_log_probs - old_log_probs.detach()) # old_log_probs are from data collection policy
            surr1 = ratio * advantages.detach() # Advantages are also based on old policy
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic Loss (Value Function Loss)
            # Target for V_theta(s_t) is G_t
            critic_loss = F.mse_loss(state_values_new, returns_g_t.detach())

            # Entropy Bonus (Part of Eq. 5)
            entropy = dist_new.entropy().mean()
            entropy_loss = -self.c2_entropy_coeff * entropy

            # Overall Loss (Eq. 5 implies L = L_actor - c1*L_vf + c2*L_entropy, but usually it's L_actor + c1*L_vf - c2*L_entropy)
            # The prompt's Eq. 5 is L = actor_loss + c1_value_loss_coeff * critic_loss + entropy_loss
            # Note: entropy_loss is already -coeff * entropy. So if Eq.5 is L = ... + S, it becomes L = ... - c2*entropy
            # If the paper's S term is H(pi), then the loss includes -c2 * H(pi). My entropy_loss is correct.
            loss = actor_loss + self.c1_value_loss_coeff * critic_loss + entropy_loss

            # Gradient Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5) # Optional grad clipping
            self.optimizer.step()

        self.clear_memory() # Clear memory after all PPO epochs for this batch are done

    def save_model(self, filepath: str):
        """Saves the actor-critic model's state dictionary."""
        torch.save(self.actor_critic.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Loads the actor-critic model's state dictionary."""
        self.actor_critic.load_state_dict(torch.load(filepath, map_location=self.device))
        self.actor_critic.to(self.device) # Ensure model is on the correct device after loading
        print(f"Model loaded from {filepath}")


if __name__ == '__main__':
    print("Testing PPOAgent...")
    state_dim = 8
    action_dim = 3
    
    # Test on CPU for simplicity in this example run
    agent = PPOAgent(state_dim, action_dim, batch_size=5, ppo_epochs=3, device='cpu') 

    print(f"Agent initialized on device: {agent.device}")
    print(f"Actor-Critic model: {agent.actor_critic}")

    # Simulate storing some transitions
    num_transitions = 10
    print(f"\nSimulating {num_transitions} transitions...")
    for i in range(num_transitions):
        s = np.random.rand(state_dim).astype(np.float32)
        # Test select_action
        if i == 0: print("Testing select_action once...")
        a, lp = agent.select_action(s)
        if i == 0: print(f"  Sample action: {a}, log_prob: {lp:.4f}")
        
        r = np.random.rand()
        s_prime = np.random.rand(state_dim).astype(np.float32)
        # Make the last transition terminal for testing _calculate_discounted_returns
        d = (i == num_transitions - 1) 
        
        agent.store_transition(s, a, lp, r, s_prime, d)

    print(f"Memory size after storing transitions: {len(agent.memory)}")
    assert len(agent.memory) == num_transitions

    # Test update
    print("\nTesting update()...")
    # Manually inspect some values before update
    # print("Sample from memory (first transition):")
    # print(f"  State: {agent.memory[0][0][:3]}... Action: {agent.memory[0][1]}, LogProb: {agent.memory[0][2]:.4f}")
    # print(f"  Reward: {agent.memory[0][3]:.4f}, NextState: {agent.memory[0][4][:3]}..., Done: {agent.memory[0][5]}")

    # Test _calculate_discounted_returns (called within update)
    # Example: rewards = [0.1, 0.2, 0.3], dones = [False, False, True], last_value = tensor(0.0) gamma = 0.99
    # R3 (terminal): 0.0
    # r2 = 0.3 -> R2 = 0.3 + 0.99 * 0 = 0.3
    # r1 = 0.2 -> R1 = 0.2 + 0.99 * 0.3 = 0.2 + 0.297 = 0.497
    # r0 = 0.1 -> R0 = 0.1 + 0.99 * 0.497 = 0.1 + 0.49203 = 0.59203
    # Expected returns: [0.59203, 0.497, 0.3]
    
    # We can't easily test _calculate_discounted_returns in isolation here without more setup,
    # but its logic is part of the update call.

    agent.update() # This will run if len(memory) >= batch_size (10 >= 5)
    print("Update() method called.")

    # Check if memory is cleared after update
    print(f"Memory size after update: {len(agent.memory)}")
    assert len(agent.memory) == 0, "Memory should be cleared after update."

    # Test save and load model
    model_filepath = "./test_ppo_agent_model.pth"
    print(f"\nTesting save_model() to {model_filepath}...")
    agent.save_model(model_filepath)
    
    print(f"Testing load_model() from {model_filepath}...")
    # Create a new agent instance to load into
    new_agent = PPOAgent(state_dim, action_dim, device='cpu')
    new_agent.load_model(model_filepath)
    print("Model loaded into new agent instance.")

    # Verify that parameters are loaded (simple check: compare a parameter)
    # This is a basic check; more rigorous checks would compare all parameters.
    original_param = next(agent.actor_critic.parameters()).clone()
    loaded_param = next(new_agent.actor_critic.parameters())
    assert torch.equal(original_param, loaded_param), "Model parameters differ after loading."
    print("Parameter check successful: loaded model matches original.")

    # Clean up the test model file
    import os
    if os.path.exists(model_filepath):
        os.remove(model_filepath)
        print(f"Cleaned up test model file: {model_filepath}")

    print("\nPPOAgent basic tests completed successfully.")
