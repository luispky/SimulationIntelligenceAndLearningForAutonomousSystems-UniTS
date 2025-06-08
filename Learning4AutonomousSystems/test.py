import gymnasium as gym
import torch
from ppo import PPOAgent


def run_test(episodes=3, save_gif=False, gif_filename="ppo_cartpole_test.gif"):
    """
    Tests the trained PPO agent, optionally saving a GIF of the policy.
    """
    # Create CartPole-v1 environment for testing
    # The render_mode will be set by the evaluate_policy function if render=True or save_gif=True
    env_test_render_mode = "rgb_array" if save_gif else "human"
    env_test = gym.make("CartPole-v1", render_mode=env_test_render_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agent and load weights
    agent = PPOAgent(
        env=env_test,
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

    try:
        agent.load("ppo_cartpole.pth")
        print("Loaded trained model ppo_cartpole.pth")
    except FileNotFoundError:
        print(
            "Error: Model file ppo_cartpole.pth not found. Train the model first using main.py."
        )
        env_test.close()
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        env_test.close()
        return

    if not save_gif:
        input("Press Enter to test the trained agent with rendering...")
        avg_test_reward = agent.evaluate_policy(
            episodes=episodes, render=True, save_gif=False
        )
        print(
            f"Average test reward over {episodes} episodes (visual): {avg_test_reward:.2f}"
        )
    else:
        print(f"Running test to save GIF ({gif_filename})...")
        avg_test_reward = agent.evaluate_policy(
            episodes=episodes, render=False, save_gif=True, gif_filename=gif_filename
        )
        print(
            f"Average test reward over {episodes} episodes (GIF saved): {avg_test_reward:.2f}"
        )

    env_test.close()


if __name__ == "__main__":
    # To visualize:
    # run_test(episodes=3, save_gif=False)

    # To save a GIF of the policy:
    run_test(episodes=1, save_gif=True, gif_filename="cartpole_policy_animation.gif")
