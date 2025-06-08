import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from ppo import PPOAgent


def main():
    # Create CartPole-v1 environment
    env = gym.make("CartPole-v1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        env=env,
        device=device,
        rollout_size=1024,
        lr=3e-4,
        epochs=10,
        gamma=0.99,
        lam=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        entropy_coef=0.01,
        batch_size=64,
    )

    print("Starting training...")
    total_timesteps = 50_000
    # log_interval for printing average rewards and storing them
    log_interval = 10
    rewards_history, avg_rewards_history = agent.train(
        total_timesteps=total_timesteps, log_interval=log_interval
    )
    print("Training finished!")

    agent.save("ppo_cartpole.pth")
    env.close()  # Close training environment

    # Plot the training rewards
    plt.figure(figsize=(12, 5))
    plt.plot(rewards_history, label="Episode Reward")
    plt.title("Episode Rewards During Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

    # Plot the average reward
    if avg_rewards_history:
        x_vals = [x[0] for x in avg_rewards_history]
        y_vals = [x[1] for x in avg_rewards_history]
        plt.figure(figsize=(12, 5))
        plt.plot(
            x_vals,
            y_vals,
            marker="o",
            label=f"Avg Reward (every {log_interval} eps)",
        )
        plt.title("Average Episode Reward During Training")
        plt.xlabel("Episode Count")
        plt.ylabel("Avg Reward")
        plt.legend()
        plt.show()

    # Test the trained agent
    env_test = gym.make("CartPole-v1", render_mode="human")
    agent.env = env_test  # Update agent's env for evaluation

    input("Press Enter to test the trained agent with rendering...")
    avg_test_reward = agent.evaluate_policy(episodes=3, render=True)
    print(f"Average test reward over 3 episodes: {avg_test_reward:.2f}")
    env_test.close()


if __name__ == "__main__":
    main()
