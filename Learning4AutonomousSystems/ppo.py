import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from nets import ActorCritic
from rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(
        self,
        env,
        device,
        rollout_size=2048,
        gamma=0.99,
        lam=0.95,
        lr=3e-4,
        epochs=10,
        batch_size=64,
        clip_coef=0.2,
        vf_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.env = env
        self.device = device

        # Basic environment details
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Model
        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Rollout buffer
        self.buffer = RolloutBuffer(rollout_size, obs_dim)

    def select_action(self, obs):
        """
        Chooses an action given a single observation (numpy array).
        Returns: action (int), log_prob (float), value (float).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def train(self, total_timesteps=100_000, log_interval=10):
        """
        Main training loop for PPO.
        """
        obs, info = self.env.reset(seed=42)

        episode_reward = 0.0
        episode_count = 0

        rewards_history = []
        avg_rewards_history = []  # Stores tuples of (episode_count, avg_reward)

        timesteps_collected_current_rollout = 0

        for current_timestep in range(total_timesteps):
            action, log_prob, value = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.store(obs, action, log_prob, value, reward, done)

            obs = next_obs
            episode_reward += reward
            timesteps_collected_current_rollout += 1

            if done:
                rewards_history.append(episode_reward)
                episode_count += 1
                if episode_count % log_interval == 0:
                    avg_reward = np.mean(rewards_history[-log_interval:])
                    avg_rewards_history.append((episode_count, avg_reward))
                    print(
                        f"Episode: {episode_count}, Timestep: {current_timestep + 1}, Avg Reward (last {log_interval} eps): {avg_reward:.2f}"
                    )

                obs, info = self.env.reset(
                    seed=42
                )  # Reset environment after episode ends
                episode_reward = 0.0

            if (
                timesteps_collected_current_rollout == self.buffer.size
                or (current_timestep + 1) == total_timesteps
            ):
                # Compute value of the last observation for advantage bootstrapping
                with torch.no_grad():
                    obs_t = (
                        torch.as_tensor(obs, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    _, last_value = self.model(obs_t)
                    last_value = (
                        last_value.item() if not done else 0.0
                    )  # if done, last_value is 0

                self.buffer.compute_returns_and_advantages(
                    last_value, self.gamma, self.lam
                )
                self.update()
                self.buffer.clear()
                timesteps_collected_current_rollout = 0

        return rewards_history, avg_rewards_history

    def update(self):
        """
        Update policy and value functions using collected rollout data.
        """
        for _ in range(self.epochs):
            for (
                obs_batch,
                actions_batch,
                old_log_probs_batch,
                _,  # We don't need values here
                advantages_batch,
                returns_batch,
            ) in self.buffer.get(self.batch_size):
                obs_batch_t = torch.as_tensor(obs_batch, dtype=torch.float32).to(
                    self.device
                )
                actions_batch_t = torch.as_tensor(actions_batch, dtype=torch.int64).to(
                    self.device
                )
                old_log_probs_batch_t = torch.as_tensor(
                    old_log_probs_batch, dtype=torch.float32
                ).to(self.device)
                advantages_batch_t = torch.as_tensor(
                    advantages_batch, dtype=torch.float32
                ).to(self.device)
                returns_batch_t = torch.as_tensor(
                    returns_batch, dtype=torch.float32
                ).to(self.device)

                # Get new log probs and values
                logits, values = self.model(obs_batch_t)
                values = values.squeeze()
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs_batch = dist.log_prob(actions_batch_t)

                # Policy loss (actor)
                prob_ratio = torch.exp(new_log_probs_batch - old_log_probs_batch_t)
                surr1 = prob_ratio * advantages_batch_t
                surr2 = (
                    torch.clamp(prob_ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    * advantages_batch_t
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (critic)
                value_loss = F.mse_loss(values, returns_batch_t)

                # Entropy loss
                entropy_loss = -dist.entropy().mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

    def evaluate_policy(
        self, episodes=5, render=False, save_gif=False, gif_filename="ppo_policy.gif"
    ):
        """
        Evaluate the current policy.
        Optionally renders the environment and saves a GIF.
        """
        frames = []
        rewards = []
        for ep in range(episodes):
            obs, info = self.env.reset(
                seed=42 + ep
            )  # Vary seed for different evaluation episodes
            done = False
            ep_reward = 0.0
            current_frames = []

            while not done:
                if render or save_gif:
                    frame = self.env.render()
                    if save_gif:
                        current_frames.append(frame)

                action, _, _ = self.select_action(
                    obs
                )  # We only need the action for evaluation
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward

            rewards.append(ep_reward)
            if save_gif and ep == 0:  # Save GIF for the first episode only if requested
                frames.extend(current_frames)

        avg_reward = np.mean(rewards)

        if save_gif and frames:
            import imageio

            # Convert frames to uint8 if they are not already (common for gym render output)
            processed_frames = []
            for frame in frames:
                if frame is not None:
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    processed_frames.append(frame)
            if processed_frames:
                imageio.mimsave(gif_filename, processed_frames, fps=30)
                print(f"Saved GIF to {gif_filename}")
            else:
                print("No frames captured to save GIF.")

        return avg_reward

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, weights_only=True))
