import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO.
    This network shares initial layers and then splits into two heads:
    one for the actor (policy) and one for the critic (value function).

    Architecture based on Fig. 2 / Section 5 of the paper:
    "Deep Reinforcement Learning for High-Frequency Trading".
    Input -> FC(32) -> LeakyReLU -> FC(16) -> LeakyReLU
           |-> Actor Head -> FC(num_actions) -> Softmax (for action probabilities)
           |-> Critic Head -> FC(1) (for state value)
    """
    def __init__(self, input_dims: int, actor_output_dims: int, critic_output_dims: int = 1):
        """
        Initializes the ActorCritic network.

        Args:
            input_dims (int): Dimensionality of the input state. (Should be 8)
            actor_output_dims (int): Dimensionality of the actor's output, 
                                     typically the number of discrete actions. (Should be 3)
            critic_output_dims (int): Dimensionality of the critic's output, 
                                      typically 1 for the state value.
        """
        super(ActorCritic, self).__init__()

        self.input_dims = input_dims
        self.actor_output_dims = actor_output_dims
        self.critic_output_dims = critic_output_dims

        # Define Fully Connected Layers (Shared Base)
        self.fc1 = nn.Linear(self.input_dims, 32)
        self.fc2 = nn.Linear(32, 16)

        # Actor Head
        self.actor_head = nn.Linear(16, self.actor_output_dims)

        # Critic Head
        self.critic_head = nn.Linear(16, self.critic_output_dims)

        # Activation Function
        # Can also be defined as a member, e.g., self.leaky_relu = nn.LeakyReLU()
        # Using F.leaky_relu in the forward pass is also common and fine.

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor. 
                                  Shape: (batch_size, input_dims)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - action_probs (torch.Tensor): Action probabilities from the actor head.
                                               Shape: (batch_size, actor_output_dims)
                - state_value (torch.Tensor): State value estimate from the critic head.
                                              Shape: (batch_size, critic_output_dims)
        """
        # Common Base
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))

        # Actor Path
        # The output of actor_head are logits. Softmax converts them to probabilities.
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic Path
        # Output is directly the state value (linear activation is implicit)
        state_value = self.critic_head(x)

        return action_probs, state_value

if __name__ == '__main__':
    print("Testing ActorCritic model...")

    # Configuration
    input_dim = 8       # As per the state space defined for TradingEnv
    num_actions = 3     # As per the action space defined for TradingEnv (Nothing, Buy, Sell)
    batch_size = 5      # Example batch size

    # Create an instance of the ActorCritic model
    model = ActorCritic(input_dims=input_dim, actor_output_dims=num_actions)
    print("\nModel Architecture:")
    print(model)

    # Create a dummy state tensor (batch_size, input_dim)
    dummy_state = torch.randn(batch_size, input_dim)
    print(f"\nDummy input state shape: {dummy_state.shape}")

    # Pass the dummy state through the model
    # model.eval() # If using dropout or batchnorm, set to eval mode for inference testing
    action_probabilities, value_estimate = model(dummy_state)

    # Print the outputs and their shapes
    print("\n--- Outputs ---")
    print("Action Probabilities (first sample in batch):", action_probabilities[0])
    print("Action Probabilities Shape:", action_probabilities.shape)
    assert action_probabilities.shape == (batch_size, num_actions)

    print("\nValue Estimate (first sample in batch):", value_estimate[0])
    print("Value Estimate Shape:", value_estimate.shape)
    assert value_estimate.shape == (batch_size, 1) # critic_output_dims is 1

    # Check that action probabilities sum to 1 for each sample in the batch
    prob_sum = action_probabilities.sum(dim=-1)
    print("\nSum of Action Probabilities (per sample in batch):", prob_sum)
    assert torch.allclose(prob_sum, torch.ones(batch_size), atol=1e-6), \
        "Action probabilities do not sum to 1 for all samples."

    print("\nActorCritic model basic tests completed successfully.")
