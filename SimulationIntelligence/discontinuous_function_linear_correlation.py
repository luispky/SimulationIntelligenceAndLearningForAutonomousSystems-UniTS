import torch
import torch.nn as nn
from generate_data import generate_data_scenario2_discontinuous_linear
from nets import (
    create_mlp,
    train_single_fidelity,
    MultiFidelityDNN,
    train_multi_fidelity,
)
from utils import plot_1D_results_scenario, plot_correlation_3D, plot_training_losses


def run_scenario2(epochs=1000, lr=1e-3, l2_reg_net_h2=1e-2, optimizer_type="lbfgs"):
    """
    Runs Scenario 2: Discontinuous function with linear correlation.
    """
    # Generate data
    xl_np, yl_np, xh_np, yh_np, x_plot_np, yh_true_np, yl_func, yh_func = (
        generate_data_scenario2_discontinuous_linear(
            n_low=38, n_high=5, noise=False, seed=42
        )
    )

    xl, yl, xh, yh = map(
        lambda x: torch.tensor(x, dtype=torch.float32).view(-1, 1),
        [xl_np, yl_np, xh_np, yh_np],
    )
    x_plot_torch = torch.tensor(x_plot_np, dtype=torch.float32).view(-1, 1)

    # Single-fidelity
    net_sf = create_mlp(
        input_dim=1, output_dim=1, hidden_units=[20] * 4, activation=nn.Tanh()
    )
    losses_sf = train_single_fidelity(
        model=net_sf,
        x=xh,
        y=yh,
        epochs=epochs,
        lr=lr,
        optimizer_type=optimizer_type,
    )
    net_sf.eval()
    with torch.no_grad():
        y_sf_pred_np = net_sf(x_plot_torch).cpu().numpy()

    # Multi-fidelity
    mf_model = MultiFidelityDNN(
        input_dims=[1, 2],
        output_dim=1,
        hidden_units_l=[20] * 4,
        hidden_units_h1=[],
        hidden_units_h2=[10] * 2,
        activation=nn.Tanh(),
    )

    losses_mf = train_multi_fidelity(
        model=mf_model,
        xl=xl,
        yl=yl,
        xh=xh,
        yh=yh,
        epochs=epochs,
        lr=lr,
        l2_reg_net_h2=l2_reg_net_h2,
        optimizer_type=optimizer_type,
    )

    mf_model.eval()
    with torch.no_grad():
        y_mf_pred_np = mf_model.predict(x_plot_torch, use_net_h1=True).cpu().numpy()
        yh_mf_pred_np = mf_model.predict(x_plot_torch, use_net_h1=False).cpu().numpy()

    # Plot results
    plot_1D_results_scenario(
        x_plot_np,
        yh_true_np,
        xh_np,
        yh_np,
        y_pred_sf=y_sf_pred_np,
        y_pred_mf=y_mf_pred_np,
        yh_pred_mf=yh_mf_pred_np,
        scenario_title="Scenario 2: Discontinuous Function + Linear Correlation",
        xl_train=xl_np,
        yl_train=yl_np,
        filename="scenario2_results",
    )

    # Plot correlation
    plot_correlation_3D(
        x_plot_torch,
        mf_model,
        scenario_title="Scenario 2",
        yl_func_exact=yl_func,
        yh_func_exact=yh_func,
        filename="scenario2_correlations",
    )

    # Plot losses
    plot_training_losses(
        losses_sf=losses_sf,
        losses_mf=losses_mf,
        title="Scenario 2",
        filename="scenario2_losses",
    )


if __name__ == "__main__":
    run_scenario2(epochs=1000, lr=1e-3, l2_reg_net_h2=1e-2, optimizer_type="adam")
