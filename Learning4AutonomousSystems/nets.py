import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """
    Simple MLP Actor-Critic for CartPole-v1.

    Observation shape: (4,)
    Action space: discrete (2)

    This model has shared hidden layers, a policy head for actions, and a value head for state values.

    The policy head outputs logits for the action space, and the value head outputs a single value for the state.

    The model is designed to be used with Proximal Policy Optimization (PPO) algorithm.

    Args:
        obs_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space (number of discrete actions).
    """

    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared hidden layers
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Policy head (actor)
        self.policy_head = nn.Linear(128, action_dim)

        # Value head (critic)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch_size, obs_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor: policy logits
        logits = self.policy_head(x)
        # Critic: state-value
        value = self.value_head(x)
        return logits, value
