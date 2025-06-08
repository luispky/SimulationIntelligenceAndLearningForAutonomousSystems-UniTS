import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def create_mlp(
    input_dim: int, output_dim: int, hidden_units: List[int], activation: nn.Module
) -> nn.Sequential:
    """
    Create a Multi-Layer Perceptron (MLP).
    """
    layers: List[nn.Module] = []
    in_dim = input_dim

    for units in hidden_units:
        linear_layer = nn.Linear(in_dim, units)
        nn.init.xavier_uniform_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)
        layers.append(linear_layer)
        layers.append(activation)
        in_dim = units

    output_linear_layer = nn.Linear(in_dim, output_dim)
    nn.init.xavier_uniform_(output_linear_layer.weight)
    nn.init.zeros_(output_linear_layer.bias)
    layers.append(output_linear_layer)

    return nn.Sequential(*layers)


class MultiFidelityDNN(nn.Module):
    """
    Multi-Fidelity Deep Neural Network.
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dim: int,
        hidden_units_l: Optional[List[int]] = None,
        hidden_units_h1: Optional[List[int]] = None,
        hidden_units_h2: Optional[List[int]] = None,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        self.input_dim_l = input_dims[0]
        self.input_dim_h = input_dims[1]

        # Low-fidelity network (Net_L)
        self.net_l = create_mlp(
            self.input_dim_l, output_dim, hidden_units_l, activation
        )

        # Linear correction network (Net_H1)
        self.net_h1 = create_mlp(
            self.input_dim_h, output_dim, hidden_units_h1, nn.Identity()
        )

        # Non-linear correction network (Net_H2)
        self.net_h2 = create_mlp(
            self.input_dim_h, output_dim, hidden_units_h2, activation
        )

    def forward_low_fidelity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the low-fidelity network.

        Parameters:
        - x: Low-fidelity input tensor.
        Returns:
        - yl_pred: Low-fidelity prediction tensor.
        """
        return self.net_l(x)

    def forward_high_fidelity(
        self, xh: torch.Tensor, yl_star: torch.Tensor, use_net_h1: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through the high-fidelity correction networks.

        Parameters:
        - xh: High-fidelity input tensor.
        - yl_star: Low-fidelity output tensor at high-fidelity inputs (net_l(xh)).
        - use_net_h1: Whether to use the linear correction network (net_h1).
        Returns:
        - yh_pred: High-fidelity prediction tensor.
        """
        x_concat = torch.cat([xh, yl_star], dim=1)
        if not use_net_h1:
            return self.net_h2(x_concat)
        return self.net_h1(x_concat) + self.net_h2(x_concat)

    def forward(
        self,
        xl: torch.Tensor,
        xh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training, returning predictions for both fidelities.

        Parameters:
        - xl: Low-fidelity input tensor.
        - xh: High-fidelity input tensor.

        Returns:
        - yl_pred: Low-fidelity prediction tensor.
        - yh_pred: High-fidelity prediction tensor.
        """
        yl_pred = self.forward_low_fidelity(xl)
        yl_pred_h = self.forward_low_fidelity(xh)
        yh_pred = self.forward_high_fidelity(xh, yl_pred_h)
        return yl_pred, yh_pred

    def predict(self, x: torch.Tensor, use_net_h1: bool = True) -> torch.Tensor:
        """
        Predict high-fidelity output for a given input.

        Args:
            x (torch.Tensor): Input tensor.
            use_net_h1 (bool): Whether to use the first high-fidelity network.

        Returns:
            torch.Tensor: High-fidelity prediction.
        """
        yl_pred = self.forward_low_fidelity(x)
        return self.forward_high_fidelity(x, yl_pred, use_net_h1)


def train_single_fidelity(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer_type: str = "lbfgs",  # 'adam' or 'lbfgs'
    epochs: int = 1000,
    lr: float = 0.01,
    max_iter_lbfgs: int = 20,
) -> List[float]:
    """
    Train a single-fidelity model using Adam or L-BFGS optimizer.

    Args:
        model (torch.nn.Module): The neural network model to train.
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Target tensor.
        optimizer_type (str, optional): Optimizer to use ('adam' or 'lbfgs'). Defaults to 'adam'.
        epochs (int, optional): Number of training epochs. Defaults to 1000.
        lr (float, optional): Learning rate. Defaults to 0.01.
        max_iter_lbfgs (int, optional): Max iterations per L-BFGS step. Defaults to 20.

    Returns:
        List[float]: List of losses per epoch.
    """
    print("\nTraining single-fidelity model\n")

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be torch.Tensor, got {type(x).__name__}")
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"y must be torch.Tensor, got {type(y).__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    criterion = torch.nn.MSELoss()

    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(), max_iter=max_iter_lbfgs, lr=lr
        )
    else:
        raise ValueError("Unsupported optimizer_type. Choose 'adam' or 'lbfgs'.")

    log_freq = epochs // 10

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()

        current_loss = 0.0
        if optimizer_type == "adam":
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
        elif optimizer_type == "lbfgs":

            def closure():
                optimizer.zero_grad()
                y_pred_ = model(x)
                loss_ = criterion(y_pred_, y)
                loss_.backward()
                return loss_

            optimizer.step(closure)
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)
                current_loss = loss.item()

        losses.append(current_loss)
        if epoch % log_freq == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}]: Loss {current_loss}")

    return losses


def train_multi_fidelity(
    model: torch.nn.Module,
    xl: torch.Tensor,
    yl: torch.Tensor,
    xh: torch.Tensor,
    yh: torch.Tensor,
    optimizer_type: str = "lbfgs",  # 'adam' or 'lbfgs'
    epochs: int = 1000,
    lr: float = 1e-3,
    l2_reg_net_h2: float = 1e-2,
) -> List[float]:
    """
    Train a multi-fidelity model using Adam or L-BFGS optimizer.

    Args:
        model (torch.nn.Module): The multi-fidelity neural network model to train.
        xl (torch.Tensor): Low-fidelity input tensor.
        yl (torch.Tensor): Low-fidelity target tensor.
        xh (torch.Tensor): High-fidelity input tensor.
        yh (torch.Tensor): High-fidelity target tensor.
        optimizer_type (str, optional): Optimizer to use ('adam' or 'lbfgs'). Defaults to 'lbfgs'.
        epochs (int, optional): Number of training epochs. Defaults to 1000.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        l2_reg_net_h2 (float, optional): L2 regularization for net_h2. Defaults to 1e-2.

    Returns:
        List[float]: List of losses per epoch.
    """
    print("\nTraining multi-fidelity model\n")
    for var, name in zip([xl, yl, xh, yh], ["xl", "yl", "xh", "yh"]):
        if not isinstance(var, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(var).__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    xl = xl.to(device)
    yl = yl.to(device)
    xh = xh.to(device)
    yh = yh.to(device)

    criterion = torch.nn.MSELoss()

    def compute_primary_loss() -> torch.Tensor:
        yl_pred, yh_pred = model(xl, xh)
        return criterion(yl_pred, yl) + criterion(yh_pred, yh)

    def compute_l2_reg() -> float:
        l2 = sum(torch.norm(param, 2) ** 2 for param in model.net_h2.parameters())
        return l2 * l2_reg_net_h2

    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            [
                {"params": model.net_l.parameters()},
                {"params": model.net_h1.parameters()},
                {"params": model.net_h2.parameters(), "weight_decay": l2_reg_net_h2},
            ],
            lr=lr,
        )
    elif optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer_type. Choose 'adam' or 'lbfgs'.")

    log_freq = epochs // 10

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()

        current_loss = 0.0
        if optimizer_type == "adam":
            optimizer.zero_grad()
            loss = compute_primary_loss()
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
        elif optimizer_type == "lbfgs":

            def closure():
                optimizer.zero_grad()
                primary_loss = compute_primary_loss()
                l2_reg = compute_l2_reg()
                total_loss = primary_loss + l2_reg
                total_loss.backward()
                return total_loss.item()

            optimizer.step(closure)
            with torch.no_grad():
                loss = compute_primary_loss() + compute_l2_reg()
                current_loss = loss.item()

        losses.append(current_loss)
        if epoch % log_freq == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}]: Loss {current_loss}")

    return losses
