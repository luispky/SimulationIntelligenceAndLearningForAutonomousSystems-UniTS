import numpy as np


class RolloutBuffer:
    """
    Stores transitions for PPO and computes GAE-based advantages.
    """

    def __init__(self, size, obs_dim):
        self.size = size

        # We store 1D observations in shape (size, obs_dim).
        # For each transition: (obs, action, log_prob, value, reward, done)
        self.observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)

        # For advantage computation
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)

        self.ptr = 0  # Current size of the buffer

    def store(self, obs, action, log_prob, value, reward, done):
        idx = self.ptr
        self.observations[idx] = obs
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """
        last_value: value prediction for the last observation of the rollout (for bootstrapping).
        gamma: discount factor
        lam: GAE-lambda factor
        """
        gae = 0.0
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = self.values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.values[step]
            )
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[step] = gae

        self.returns = self.values + self.advantages

    def clear(self):
        self.ptr = 0

    def get(self, batch_size):
        """
        Yields mini-batches of the entire buffer in random order.
        """
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                self.observations[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.values[batch_idx],
                self.advantages[batch_idx],
                self.returns[batch_idx],
            )
