import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Callable
import torch
import os
from nets import MultiFidelityDNN


def plot_1D_results_scenario(
    x_plot: np.ndarray,
    yh_true: np.ndarray,
    xh_train: Optional[np.ndarray] = None,
    yh_train: Optional[np.ndarray] = None,
    y_pred_sf: Optional[np.ndarray] = None,
    y_pred_mf: Optional[np.ndarray] = None,
    yh_pred_mf: Optional[np.ndarray] = None,
    scenario_title: str = "Scenario Results",
    xl_train: Optional[np.ndarray] = None,
    yl_train: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
) -> None:
    """Plots 1D function approximation results for a given scenario."""
    plt.figure(figsize=(10, 6))

    # Plot true high-fidelity function
    plt.plot(x_plot, yh_true, "k-", label="True $y_h(x)$", linewidth=2)

    # Plot low-fidelity training data if available
    if xl_train is not None and yl_train is not None:
        plt.scatter(
            xl_train,
            yl_train,
            color="gray",
            marker="x",
            s=50,
            label="Low-fidelity $y_l$ data",
        )

    # Plot high-fidelity training data if availablez
    if xh_train is not None and yh_train is not None:
        plt.scatter(
            xh_train,
            yh_train,
            color="red",
            marker="o",
            s=80,
            edgecolor="black",
            label="High-fidelity $y_h$ data",
        )

    # Plot single-fidelity prediction at high-fidelity points if available
    if y_pred_sf is not None:
        plt.plot(x_plot, y_pred_sf, "--", label="SF Pred", color="magenta")
    # Plot multi-fidelity prediction (with H1) if available
    if y_pred_mf is not None:
        plt.plot(x_plot, y_pred_mf, "--", label="MF Pred", color="red")
    # Plot multi-fidelity prediction (without H1) if available
    if yh_pred_mf is not None:
        plt.plot(x_plot, yh_pred_mf, "--", label="MF Pred @ HF", color="blue")

    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.title(scenario_title, fontsize=15)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if filename:
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()


def plot_correlation_3D(
    x_plot_torch: torch.Tensor,
    mf_model: MultiFidelityDNN,
    scenario_title: str,
    yl_func_exact: Callable[[np.ndarray], np.ndarray],
    yh_func_exact: Callable[[np.ndarray], np.ndarray],
    filename: Optional[str] = None,
) -> None:
    """Plots the 3D correlation learned by the Multi-Fidelity model."""
    mf_model.eval()
    with torch.no_grad():
        # Predicted low-fidelity
        yl_star = mf_model.forward_low_fidelity(x_plot_torch)
        # Predicted high-fidelity
        yh_star = mf_model.forward_high_fidelity(x_plot_torch, yl_star)

    x_np = x_plot_torch.cpu().numpy().flatten()
    yl_np = yl_star.cpu().numpy().flatten()
    yh_np = yh_star.cpu().numpy().flatten()

    # If exact correlation is known:
    plot_exact = (yl_func_exact is not None) and (yh_func_exact is not None)
    if plot_exact:
        yl_exact = yl_func_exact(x_np)
        yh_exact = yh_func_exact(x_np)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 3D Predicted Curve
    ax.plot(yl_np, yh_np, x_np, "r--", label="MF Pred", linewidth=2)

    # 3D Exact Curve
    if plot_exact:
        ax.plot(
            yl_exact.flatten(),
            yh_exact.flatten(),
            x_np,
            "k-",
            label="Exact",
            linewidth=2,
        )

    # 2D Predicted Curve projected onto z=0 plane
    ax.plot(yl_np, yh_np, zs=0, zdir="z", label="2D Pred", color="magenta", linewidth=2)

    # 2D Exact Curve projected onto z=0 plane
    if plot_exact:
        ax.plot(
            yl_exact.flatten(),
            yh_exact.flatten(),
            zs=0,
            zdir="z",
            label="2D Exact",
            color="blue",
            linewidth=2,
        )

    # Set labels
    ax.set_xlabel("yl", fontsize=12, labelpad=10)
    ax.set_ylabel("yh", fontsize=12, labelpad=10)
    ax.set_zlabel("x", fontsize=12, labelpad=10)
    ax.set_title(f"{scenario_title}: Learned vs True Correlation", fontsize=15, pad=20)
    ax.legend(fontsize=10)
    ax.view_init(elev=20, azim=-65)  # Adjust view angle for better visualization
    # Add legend
    ax.legend()
    plt.tight_layout()

    ax.grid(True)
    if filename:
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()


def plot_training_losses(
    losses_sf: Optional[List[float]] = None,
    losses_mf: Optional[List[float]] = None,
    title: str = "Training Loss Comparison",
    filename: Optional[str] = None,
) -> None:
    """Plots training losses for single and multi-fidelity models."""
    plt.figure(figsize=(8, 5))

    plot_exists = False
    if losses_sf:
        plt.plot(losses_sf, label="Single-Fidelity Loss", color="blue")
        plot_exists = True
    if losses_mf:
        plt.plot(losses_mf, label="Multi-Fidelity Loss", color="green")
        plot_exists = True

    if not plot_exists:
        print("No loss data provided to plot_training_losses.")
        plt.close()  # Close the empty figure
        return

    plt.xlabel("Epoch/Iteration Step", fontsize=12)
    # Determine if log scale is appropriate
    min_loss_sf = min(losses_sf) if losses_sf else float("inf")
    min_loss_mf = min(losses_mf) if losses_mf else float("inf")
    max_loss_sf = max(losses_sf) if losses_sf else float("-inf")
    max_loss_mf = max(losses_mf) if losses_mf else float("-inf")

    min_overall_loss = min(min_loss_sf, min_loss_mf)
    max_overall_loss = max(max_loss_sf, max_loss_mf)

    if (
        min_overall_loss > 0 and max_overall_loss / min_overall_loss > 100
    ):  # Heuristic for using log scale
        plt.yscale("log")
        plt.ylabel("Loss (log-scale)", fontsize=12)
    else:
        plt.ylabel("Loss", fontsize=12)

    plt.title(f"{title}: Training Losses", fontsize=15)
    if losses_sf or losses_mf:
        plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if filename:
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()
