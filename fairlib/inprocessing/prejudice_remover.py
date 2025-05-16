import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Callable, Union, Any
from typing_extensions import override
from fairlib import DataFrame, logger
from .in_processing import Processor


def _convert_to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, DataFrame):
        return torch.tensor(x.to_numpy().astype(float)).float()
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x.astype(float), dtype=torch.float32)
    return x


class PrejudiceRemoverLoss:
    def __init__(
        self,
        base_loss_fn: Callable = nn.BCELoss(),
        eta: float = 1.0,
    ):
        """
        Initialize the PrejudiceRemoverLoss.

        Args:
            base_loss_fn: Base loss function (default: BCELoss)
            eta: Fairness regularization parameter (default: 1.0)
        """
        self.base_loss_fn = base_loss_fn
        self.eta = eta
        self.__sensitive_feature = None

    @property
    def sensitive_feature(self):
        """
        Returns the sensitive feature used for the penalty term.

        Returns:
            The sensitive feature used for the penalty term.
        """
        return self.__sensitive_feature

    @sensitive_feature.setter
    def sensitive_feature(self, value):
        """
        Sets the sensitive feature used for the penalty term.

        Args:
            value: The sensitive feature.
        """
        self.__sensitive_feature = value

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the prejudice remover loss.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Total loss with prejudice regularization
        """
        # Base loss
        base_loss = self.base_loss_fn(y_pred, y_true)

        # If no regularization or no sensitive feature, return base loss
        if self.eta == 0.0 or self.sensitive_feature is None:
            return base_loss

        # Calculate mutual information regularization term
        # This implements the prejudice regularizer from the paper
        s = self.sensitive_feature

        # Calculate P(y=1|s=1) and P(y=1|s=0)
        s_pos_mask = (s == 1).float()
        s_neg_mask = (s == 0).float()

        # Avoid division by zero
        s_pos_count = torch.sum(s_pos_mask) + 1e-8
        s_neg_count = torch.sum(s_neg_mask) + 1e-8

        # Calculate conditional probabilities
        p_y1_s1 = torch.sum(y_pred * s_pos_mask) / s_pos_count
        p_y1_s0 = torch.sum(y_pred * s_neg_mask) / s_neg_count

        # Calculate P(y=0|s=1) and P(y=0|s=0)
        p_y0_s1 = torch.sum((1 - y_pred) * s_pos_mask) / s_pos_count
        p_y0_s0 = torch.sum((1 - y_pred) * s_neg_mask) / s_neg_count

        # Calculate P(s=1) and P(s=0)
        p_s1 = s_pos_count / (s_pos_count + s_neg_count)
        p_s0 = s_neg_count / (s_pos_count + s_neg_count)

        # Calculate P(y=1) and P(y=0)
        p_y1 = torch.mean(y_pred)
        p_y0 = 1 - p_y1

        # Calculate mutual information regularization term
        # MI = sum_{y,s} P(y,s) * log(P(y,s) / (P(y) * P(s)))
        # We use KL divergence as a measure of mutual information

        # Calculate joint probabilities
        p_y1_s1_joint = p_y1_s1 * p_s1
        p_y1_s0_joint = p_y1_s0 * p_s0
        p_y0_s1_joint = p_y0_s1 * p_s1
        p_y0_s0_joint = p_y0_s0 * p_s0

        # Calculate products of marginals
        p_y1_p_s1 = p_y1 * p_s1
        p_y1_p_s0 = p_y1 * p_s0
        p_y0_p_s1 = p_y0 * p_s1
        p_y0_p_s0 = p_y0 * p_s0

        # Calculate KL terms (with small epsilon to avoid log(0))
        epsilon = 1e-8
        kl_y1_s1 = p_y1_s1_joint * torch.log(
            (p_y1_s1_joint + epsilon) / (p_y1_p_s1 + epsilon)
        )
        kl_y1_s0 = p_y1_s0_joint * torch.log(
            (p_y1_s0_joint + epsilon) / (p_y1_p_s0 + epsilon)
        )
        kl_y0_s1 = p_y0_s1_joint * torch.log(
            (p_y0_s1_joint + epsilon) / (p_y0_p_s1 + epsilon)
        )
        kl_y0_s0 = p_y0_s0_joint * torch.log(
            (p_y0_s0_joint + epsilon) / (p_y0_p_s0 + epsilon)
        )

        # Sum all KL terms to get mutual information
        mutual_info = kl_y1_s1 + kl_y1_s0 + kl_y0_s1 + kl_y0_s0

        # Total loss with prejudice regularization
        total_loss = base_loss + self.eta * mutual_info

        return total_loss


class PrejudiceRemover(Processor):
    """
    Implementation of the Prejudice Remover algorithm.

    This algorithm learns a classifier while removing direct and indirect prejudice
    by adding a regularization term that reduces the mutual information between
    the sensitive attribute and the prediction.
    """

    def __init__(
        self,
        torchModel: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: nn.Module = nn.BCELoss(),
        eta: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the PrejudiceRemover model.

        Args:
            torchModel: PyTorch model to train
            optimizer: PyTorch optimizer (default: Adam)
            loss: Base loss function (default: BCELoss)
            eta: Fairness regularization parameter (default: 1.0)
            **kwargs: Additional parameters for model compilation
        """
        if not isinstance(torchModel, nn.Module):
            raise TypeError(f"Expected a Torch model, got {type(torchModel)}")
        super().__init__(torchModel)
        self.model = torchModel
        self.__compilation_parameters = kwargs
        self.optimizer = optimizer or optim.Adam(torchModel.parameters())
        self.base_loss = loss
        self.eta = eta
        self.total_loss = PrejudiceRemoverLoss(
            base_loss_fn=self.base_loss,
            eta=self.eta,
        )

    def fit(self, x: DataFrame, y: Optional[Any] = None, **kwargs):
        """
        Train the PrejudiceRemover model.

        Args:
            x: Input DataFrame with sensitive attributes
            y: Target labels (if not included in x)
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Trained model instance
        """
        if not isinstance(x, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(x)}")
        if y is None:
            x, y, _, _, _, sensitive_indexes = x.unpack()
        else:
            x, _, _, _, _, sensitive_indexes = x.unpack()
        if len(sensitive_indexes) != 1:
            raise ValueError(
                f"PrejudiceRemover expects exactly one sensitive column, got {len(sensitive_indexes)}: {x.sensitive}"
            )
        sensitive_attr = x[:, sensitive_indexes[0]]

        # Extract hyperparameters
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 32)

        # Ensure all inputs are tensors and have the same dtype
        x = _convert_to_tensor(x)
        y = _convert_to_tensor(y)
        sensitive_attr = _convert_to_tensor(sensitive_attr)

        # Prepare dataset
        if sensitive_attr is not None:
            dataset = torch.utils.data.TensorDataset(x, y, sensitive_attr)
        else:
            dataset = torch.utils.data.TensorDataset(x, y)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                if sensitive_attr is not None:
                    batch_X, batch_y, batch_sensitive = batch
                else:
                    batch_X, batch_y = batch
                    batch_sensitive = None

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)

                # Compute loss
                self.total_loss.sensitive_feature = batch_sensitive
                loss = self.total_loss(batch_y, outputs)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}"
                )

        return self

    def predict(self, x: DataFrame, **kwargs) -> torch.Tensor:
        """
        Make predictions using the trained model.

        Args:
            x (DataFrame): Input features for prediction
            batch_size (int): Batch size for prediction
        Returns:
            torch.Tensor: Model predictions
        """

        if not isinstance(x, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(x)}")

        # Extract hyperparameters
        batch_size = kwargs.get("batch_size", 32)
        # Ensure model is in evaluation mode
        self.model.eval()
        x = _convert_to_tensor(x)
        # Prepare prediction dataset
        pred_dataset = torch.utils.data.TensorDataset(x)
        pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=batch_size)

        # Collect predictions
        predictions = []

        # Disable gradient computation for inference
        with torch.no_grad():
            for (batch_X,) in pred_loader:
                # Forward pass
                batch_pred = self.model(batch_X)

                # Store predictions
                predictions.append(batch_pred.cpu())

        # Concatenate predictions
        return torch.cat(predictions, dim=0)
