import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pandas import Series
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Union, Any
from .in_processing import Processor
from fairlib import logger, DataFrame


def _convert_to_tensor(x: Any, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, (DataFrame, Series)):
        return torch.tensor(x.to_numpy(), dtype=dtype)
    if not isinstance(x, torch.Tensor):
        return torch.tensor(np.asarray(x), dtype=dtype)
    return x


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    Forwards input as-is, but reverses gradients during backward pass.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Forward pass stores lambda_ value and returns input unchanged.
        Args:
            x: Input tensor
            lambda_: Gradient scaling factor
        Returns:
            Input tensor unchanged
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass reverses gradient and scales by lambda_.
        Args:
            grad_output: Gradient from subsequent layer
        Returns:
            Tuple of (reversed & scaled gradient, None)
        """
        grad_output = grad_outputs[0]
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Helper function for gradient reversal."""
    return GradientReversalFunction.apply(x, lambda_)


class Predictor(nn.Module):
    """
    Main prediction network that learns to predict target labels
    while trying to be fair with respect to sensitive attributes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.3,
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (number of classes)
            dropout_rate: Dropout probability for regularization
        """
        super(Predictor, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, x: torch.Tensor, return_representation: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout(x)
        rep = F.relu(self.fc2(x))
        rep = self.bn3(rep)
        rep = self.dropout(rep)
        logits = self.fc3(rep).squeeze()
        return (logits, rep) if return_representation else logits


class Adversary(nn.Module):
    """
    Adversarial network that tries to predict sensitive attributes
    from the predictor's internal representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sensitive_dim: int,
        dropout_rate: float = 0.3,
    ):
        """
        Args:
            input_dim: Dimension of input features (predictor's representation)
            hidden_dim: Dimension of hidden layers
            sensitive_dim: Dimension of sensitive attribute prediction
            dropout_rate: Dropout probability for regularization
        """
        super(Adversary, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, sensitive_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout(x)
        return self.fc3(x)


class AdversarialDebiasing(nn.Module, Processor):
    """
    Complete adversarial debiasing model that combines predictor and adversary.

    Recommended lambda_adv values:
    - 0.0: No debiasing (baseline model)
    - 0.1-0.5: Weak debiasing
    - 1.0-2.0: Moderate debiasing (good starting point)
    - 3.0-5.0: Strong debiasing
    - >5.0: Very strong debiasing (might hurt task performance)

    Choose based on your fairness vs. performance trade-off requirements.
    """

    def __init__(
        self,
        input_dim: Union[int, None] = None,
        hidden_dim: Union[int, None] = None,
        output_dim: Union[int, None] = None,
        sensitive_dim: Union[int, None] = None,
        predictor: Union[Predictor, None] = None,
        adversary: Union[Adversary, None] = None,
        lambda_adv: float = 1.0,
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (number of classes)
            sensitive_dim: Dimension of sensitive attribute
            predictor: Main prediction network
            adversary: Adversarial network
            lambda_adv: Weight of adversarial loss (default=1.0)
        """
        super(AdversarialDebiasing, self).__init__()
        if predictor is None and adversary is None:
            if (
                input_dim is None
                or hidden_dim is None
                or output_dim is None
                or sensitive_dim is None
            ):
                raise ValueError(
                    "input_dim, hidden_dim, output_dim, and sensitive_dim must be provided"
                )
            self.predictor = Predictor(input_dim, hidden_dim, output_dim)
            self.adversary = Adversary(hidden_dim, hidden_dim, sensitive_dim)
        elif predictor is None or adversary is None:
            raise ValueError("Both predictor and adversary must be provided or None")
        else:
            self.predictor = predictor
            self.adversary = adversary
        self.lambda_adv = lambda_adv

    def forward(
        self, x: torch.Tensor, return_representation: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.predictor(x, return_representation=return_representation)

    def forward_adversary(self, rep: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal and adversary to input representation."""
        rep_rev = grad_reverse(rep, self.lambda_adv)
        return self.adversary(rep_rev)

    def fit(self, x: DataFrame, y: Optional[Series | np.ndarray] = None, **kwargs):
        """
        Train the model using adversarial debiasing.

        Args:
            x: Input features
            y: Target labels
            num_epochs: Number of training epochs
            lr: Learning rate
            adv_steps: Number of adversary updates per predictor update
            batch_size: Batch size for training
        Returns:
            dict: Training history
        """

        if not isinstance(x, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(x)}")
        if y is None:
            x, y, _, _, _, sensitive_indexes = x.unpack()
        else:
            x, _, _, _, _, sensitive_indexes = x.unpack()
        if len(sensitive_indexes) != 1:
            raise ValueError(
                f"Adversarial Debiasing expects exactly one sensitive column, got {len(sensitive_indexes)}: {x.sensitive}"
            )
        sensitive_attr = x[:, sensitive_indexes[0]]

        # Extract hyperparameters
        num_epochs = kwargs.get("num_epochs", 100)
        lr = kwargs.get("lr", 0.001)
        adv_steps = kwargs.get("adv_steps", 1)
        batch_size = kwargs.get("batch_size", 32)

        # Ensure all inputs are tensors and have the same dtype
        x = _convert_to_tensor(x)
        y = _convert_to_tensor(y)
        sensitive_attr = _convert_to_tensor(sensitive_attr)

        # Prepare dataset
        if sensitive_attr is not None:
            dataset = TensorDataset(x, y, sensitive_attr)
        else:
            dataset = TensorDataset(x, y)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Initialize optimizers and schedulers
        optimizer_pred = optim.AdamW(
            self.predictor.parameters(), lr=lr, weight_decay=0.01
        )
        optimizer_adv = optim.AdamW(
            self.adversary.parameters(), lr=lr, weight_decay=0.01
        )

        criterion_task = nn.BCEWithLogitsLoss()
        criterion_adv = nn.BCEWithLogitsLoss()

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "adv_loss": [],
        }

        for epoch in range(num_epochs):
            self.train()
            total_loss_task, total_loss_adv = 0.0, 0.0
            total_correct, total_samples = 0, 0

            for x_batch, y_batch, a_batch in dataloader:
                x_batch, y_batch, a_batch = (
                    x_batch.to(device),
                    y_batch.to(device),
                    a_batch.to(device),
                )

                # Only train adversary if lambda_adv > 0
                if self.lambda_adv > 0:
                    # Update Adversary
                    for _ in range(adv_steps):
                        optimizer_adv.zero_grad()
                        with torch.no_grad():
                            _, rep = self.forward(x_batch, return_representation=True)
                        assert (
                            rep is not None
                        ), "Representation must not be None when return_representation is True"
                        adv_logits = self.adversary(rep.detach()).squeeze()
                        loss_adv = criterion_adv(adv_logits, a_batch.float())
                        loss_adv.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.adversary.parameters(), max_norm=1.0
                        )
                        optimizer_adv.step()

                # Update Predictor
                optimizer_pred.zero_grad()
                pred_logits, rep = self.forward(x_batch, return_representation=True)
                loss_task = criterion_task(pred_logits, y_batch)

                if self.lambda_adv > 0:
                    assert rep is not None
                    adv_logits_for_pred = self.forward_adversary(rep).squeeze()
                    loss_adv_for_pred = criterion_adv(
                        adv_logits_for_pred, a_batch.float()
                    )
                    loss_combined = loss_task + self.lambda_adv * loss_adv_for_pred
                else:
                    loss_combined = loss_task
                    loss_adv_for_pred = torch.tensor(0.0)

                loss_combined.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.predictor.parameters(), max_norm=1.0
                )
                optimizer_pred.step()

                total_loss_task += loss_task.item()
                total_loss_adv += loss_adv_for_pred.item()
                preds = (torch.sigmoid(pred_logits) > 0.5).long()
                total_correct += (preds == y_batch).sum().item()
                total_samples += y_batch.size(0)

            train_loss = total_loss_task / len(dataloader)
            train_acc = total_correct / total_samples
            adv_loss = total_loss_adv / len(dataloader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["adv_loss"].append(adv_loss)

            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Adv Loss: {adv_loss:.4f}"
            )

        return history

    def predict(self, x: DataFrame, **kwargs) -> torch.Tensor:
        """
        Predict the target labels for input tensor x.
        This function sets the model to evaluation mode,
        performs a forward pass, and returns the predicted class indices.
        """
        if not isinstance(x, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(x)}")
        x = _convert_to_tensor(x)
        self.eval()
        with torch.no_grad():
            # Call predictor directly to get logits only
            logits = self.predictor(x, return_representation=False)
            preds = (torch.sigmoid(logits) > 0.5).long()
        return preds
