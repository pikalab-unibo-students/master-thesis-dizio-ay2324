import torch
import torch.nn as nn
import torch.optim as optim
from ._torch_metrics import get as get_metric
from typing import Optional, Callable, Union, Any
from fairlib import DataFrame, logger
from .in_processing import Processor


def _convert_to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, DataFrame):
        return torch.tensor(x.to_numpy().astype(float)).float()
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x.astype(float), dtype=torch.float32)
    return x


class BaseLoss:
    def __init__(self, base_loss_fn: Callable):
        self.base_loss_fn = base_loss_fn

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.base_loss_fn(y_pred, y_true)


class PenalizedLoss(BaseLoss):
    def __init__(
        self,
        base_loss_fn: Callable = nn.MSELoss(),
        regularizer: str = "spd",
        weight: float = 0.0,
    ):
        super().__init__(base_loss_fn)
        self.regularizer = regularizer
        self.weight = weight
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
        base_loss = super().__call__(y_true, y_pred)

        regularizer_fn = get_metric(self.regularizer)
        if regularizer_fn is None:
            raise ValueError(f"Invalid fairness metric: {self.regularizer}")

        regularizer_loss = regularizer_fn(y_true, y_pred, self.sensitive_feature)

        total_loss = (1 - self.weight) * base_loss + self.weight * regularizer_loss

        return total_loss


class Fauci(Processor):
    def __init__(
        self,
        torchModel: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: Union[nn.Module] = nn.MSELoss(),
        fairness_regularization: str = "spd",
        regularization_weight: float = 0.5,
        **kwargs,
    ):
        if not isinstance(torchModel, nn.Module):
            raise TypeError(f"Expected a Torch model, got {type(torchModel)}")
        super().__init__(torchModel)
        self.model = torchModel
        self.__compilation_parameters = kwargs
        self.optimizer = optimizer or optim.Adam(torchModel.parameters())
        self.loss = loss
        if fairness_regularization is None or regularization_weight == 0.0:
            self.total_loss = BaseLoss(base_loss_fn=self.loss)
        else:
            self.total_loss = PenalizedLoss(
                base_loss_fn=self.loss,
                regularizer=fairness_regularization,
                weight=regularization_weight,
            )

    def fit(self, x: DataFrame, y: Optional[Any] = None, **kwargs):
        if not isinstance(x, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(x)}")
        if y is None:
            x, y, _, _, _, sensitive_indexes = x.unpack()
        else:
            x, _, _, _, _, sensitive_indexes = x.unpack()
        if len(sensitive_indexes) != 1:
            raise ValueError(
                f"FaUCI expects exactly one sensitive column, got {len(sensitive_indexes)}: {x.sensitive}"
            )
        sensitive_attr = x[:, sensitive_indexes[0]]

        # Extract hyperparameters from kwargs
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
                if isinstance(self.total_loss, PenalizedLoss):
                    self.total_loss.sensitive_feature = batch_sensitive
                loss = self.total_loss(batch_y, outputs)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Print progress
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}"
                )

        return self

    def predict(self, x: DataFrame | torch.Tensor, **kwargs):
        """
        Make predictions using the trained model.

        Args:
            x (DataFrame): Input features for prediction
            batch_size (int): Batch size for prediction

        Returns:
            torch.Tensor: Model predictions
        """

        if not isinstance(x, (DataFrame | torch.Tensor)):
            raise TypeError(f"Expected a DataFrame, got {type(x)}")

        # Extract hyperparameters from kwargs
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
