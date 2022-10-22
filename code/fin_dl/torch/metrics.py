import torch
import typing as t
from abc import ABC
from torchmetrics import Metric
from typing import Any

from fin_dl.torch import TENSOR_DTYPE


def check_shape(tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
    if tensor1.size() != tensor2.size():
        raise ValueError(
            "\n".join(
                [f"Shape mismatch! {tensor1.shape} != {tensor2.shape}" f"tensor1={tensor1}", f"tensor2={tensor2}"]
            )
        )


def cast(pred: torch.Tensor, target: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    return getattr(pred, TENSOR_DTYPE)().to(pred.device), getattr(target, TENSOR_DTYPE)().to(target.device)


class R2(Metric, ABC):
    """
    R2 = 1 - SSR / SST
    SSR = sum((target - pred)**2)
    SST = sum((target - mean(target))**2) = sum(target**2) - 1 / len(target) * sum(target)**2

    Example:

    import numpy as np

    target = np.random.random(100)
    pred = np.random.random(100)

    1 - np.sum((pred - target) ** 2) / np.sum((target - np.mean(target)) ** 2)

    # is equivalent to:

    1 - np.sum((pred - target) ** 2) / (np.sum(target**2) - 1 / len(target) * np.sum(target) ** 2)
    """

    number_of_outputs: int
    full_state_update: bool = False
    higher_is_better: bool = True

    def __init__(self, number_of_outputs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # init attributes
        self.number_of_outputs = number_of_outputs
        # init states
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # cast tensors
        pred, target = cast(pred, target)
        check_shape(pred, target)

        # update states
        self.sum_of_squared_residuals += torch.sum(torch.square(pred - target), dim=0)
        self.sum_of_squared_target += torch.sum(torch.square(target), dim=0)
        self.sum_of_target += torch.sum(target, dim=0)
        self.len_of_target += target.size(0)

    def compute(self) -> Any:
        return 1 - self.sum_of_squared_residuals / (
            self.sum_of_squared_target - 1 / self.len_of_target * self.sum_of_target**2
        )

    def reset(self) -> None:
        self.add_state("sum_of_squared_residuals", default=torch.zeros(self.number_of_outputs), dist_reduce_fx="sum")
        self.add_state("sum_of_squared_target", default=torch.zeros(self.number_of_outputs), dist_reduce_fx="sum")
        self.add_state("sum_of_target", default=torch.zeros(self.number_of_outputs), dist_reduce_fx="sum")
        self.add_state("len_of_target", default=torch.zeros(1), dist_reduce_fx="sum")


class NaiveR2(Metric, ABC):
    """
    R2 = 1 - SSR / SST
    SSR = sum((target - pred)**2)
    SST = sum(target**2)
    """

    number_of_outputs: int
    full_state_update: bool = False
    higher_is_better: bool = True

    def __init__(self, number_of_outputs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # init attributes
        self.number_of_outputs = number_of_outputs

        # init states
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # cast tensors
        pred, target = cast(pred, target)
        check_shape(pred, target)

        # update states
        self.sum_of_squared_residuals += torch.sum(torch.square(pred - target), dim=0)
        self.sum_of_squared_target += torch.sum(torch.square(target), dim=0)

    def compute(self) -> Any:
        return 1 - self.sum_of_squared_residuals / self.sum_of_squared_target

    def reset(self) -> None:
        self.add_state("sum_of_squared_residuals", default=torch.zeros(self.number_of_outputs), dist_reduce_fx="sum")
        self.add_state("sum_of_squared_target", default=torch.zeros(self.number_of_outputs), dist_reduce_fx="sum")


class PearsonCorr(Metric):
    """
    corr = sum((target - mean(target)) * (pred - mean(pred))) / sqrt(
    sum((target - mean(target)) ** 2) * sum((pred - mean(pred)) ** 2)
    )

    is equivalent to:

    (np.sum(target * pred) - np.sum(target) * 1 / len(pred) * np.sum(pred)) / np.sqrt(
        (np.sum(target**2) - 1 / len(target) * np.sum(target) ** 2)
        * (np.sum(pred**2) - 1 / len(pred) * np.sum(pred) ** 2)
    )

    Example (simple):

    import numpy as np

    target = np.random.random(100)
    pred = np.random.random(100)

    corr = np.corrcoef(target, pred)[0, 1]

    # is equivalent to:

    corr = (np.sum(target * pred) - np.sum(target) * 1 / len(pred) * np.sum(pred)) / (np.sqrt((np.sum(target
    ** 2) - 1 / len(target) * np.sum(target) ** 2)) * np.sqrt((np.sum(pred ** 2) - 1 / len(pred) * np.sum(pred) ** 2)))

    Example (extended):

    # computes the correlation between each column of the target matrix and each column of the prediction matrix.

    import numpy as np

    target = np.random.random((10, 10))
    pred = np.random.random((10, 10))

    sum_of_squared_target = np.tile(np.sum(np.square(target), axis=0).reshape(target.shape[-1], -1), pred.shape[-1])
    sum_of_target = np.tile(np.sum(target, axis=0).reshape(target.shape[-1], -1), pred.shape[-1])
    len_of_target = target.shape[0]
    sum_of_squared_pred = np.tile(np.sum(np.square(pred), axis=0), target.shape[-1]).reshape(-1, pred.shape[-1])
    sum_of_pred = np.tile(np.sum(pred, axis=0), target.shape[-1]).reshape(-1, pred.shape[-1])
    len_of_pred = pred.shape[0]
    sum_of_target_times_pred = target.T @ pred

    numerator: torch.Tensor = sum_of_target_times_pred - sum_of_target * 1 / len_of_pred * sum_of_pred
    denominator: torch.Tensor = (np.sqrt(
        (sum_of_squared_target - 1 / len_of_target * sum_of_target**2))
        * np.sqrt((sum_of_squared_pred - 1 / len_of_pred * sum_of_pred**2)
    ))

    # corr: rows=>target columns | columns=>pred columns
    corr = numerator / denominator

    # check equivalence to plain for-loop calculation
    for i in range(target.shape[-1]):
        for j in range(pred.shape[-1]):
            print(np.isclose(corr[i, j], np.corrcoef(target[:, i], pred[:, j])[0, 1]))
    """

    pred_input_size: int
    target_input_size: int

    full_state_update: bool = False
    # actually higher abs(value) is better
    higher_is_better: bool = True

    def __init__(self, pred_input_size: int = 1, target_input_size: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # init attributes
        self.pred_input_size = pred_input_size
        self.target_input_size = target_input_size
        # init states
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # cast tensors
        pred, target = cast(pred, target)
        check_shape(pred, target)

        # update states
        self.sum_of_squared_target += (
            torch.sum(torch.square(target), dim=0).view(target.size(-1), -1).tile(pred.size(-1))
        )
        self.sum_of_target += torch.sum(target, dim=0).view(target.size(-1), -1).tile(pred.size(-1))
        self.len_of_target += target.size(0)
        self.sum_of_squared_pred += torch.sum(torch.square(pred), dim=0).tile(target.size(-1)).view(-1, pred.size(-1))
        self.sum_of_pred += torch.sum(pred, dim=0).tile(target.size(-1)).view(-1, pred.size(-1))
        self.len_of_pred += pred.size(0)
        self.sum_of_target_times_pred += target.T @ pred

    def compute(self) -> Any:
        numerator: torch.Tensor = (
            self.sum_of_target_times_pred - self.sum_of_target * 1 / self.len_of_pred * self.sum_of_pred
        )
        denominator: torch.Tensor = torch.multiply(
            torch.sqrt(self.sum_of_squared_target - 1 / self.len_of_target * self.sum_of_target**2),
            torch.sqrt(self.sum_of_squared_pred - 1 / self.len_of_pred * self.sum_of_pred**2),
        )
        return numerator / denominator

    def reset(self) -> None:
        self.add_state(
            "sum_of_squared_target",
            default=torch.zeros((self.target_input_size, self.pred_input_size)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_of_target", default=torch.zeros((self.target_input_size, self.pred_input_size)), dist_reduce_fx="sum"
        )
        self.add_state("len_of_target", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state(
            "sum_of_squared_pred",
            default=torch.zeros((self.target_input_size, self.pred_input_size)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_of_pred", default=torch.zeros((self.target_input_size, self.pred_input_size)), dist_reduce_fx="sum"
        )
        self.add_state("len_of_pred", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state(
            "sum_of_target_times_pred",
            default=torch.zeros((self.target_input_size, self.pred_input_size)),
            dist_reduce_fx="sum",
        )
