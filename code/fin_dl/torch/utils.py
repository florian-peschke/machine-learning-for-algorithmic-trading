import contextlib
import logging
import os
import re
import typing as t

import numpy as np
import pandas as pd
import torch
from attr import define
from lets_plot import aes, facet_grid, geom_line, geom_point, ggplot, ggsave, ggsize, labs
from lets_plot.plot.core import PlotSpec
from torch import nn
from torch.nn.parameter import Parameter
from torchmetrics import Metric
from typeguard import typechecked

from fin_dl import GGPLOT_THEME
from fin_dl.utilities import float_to_str


@define
class TimeSeriesSplit:
    inputs: pd.DataFrame
    target: pd.DataFrame
    date_label: str
    group_key: str
    splitting_proportions: t.Tuple[float, float, float]

    @typechecked
    def __init__(
        self,
        inputs: pd.DataFrame,
        target: pd.DataFrame,
        date_label: str,
        group_key: str,
        splitting_proportions: t.Tuple[float, float, float],
    ):
        self.__attrs_init__(
            inputs=inputs,
            target=target,
            date_label=date_label,
            group_key=group_key,
            splitting_proportions=splitting_proportions,
        )

    def __attrs_post_init__(self) -> None:
        check_len_eq(self.inputs, self.target)

    def _dataset(
        self, splitting_dates: t.Tuple[t.Optional[float], t.Optional[float]]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:

        inputs: pd.DataFrame = to_frame(
            pd.concat(list(self._split_data(frame=self.inputs, splitting_dates=splitting_dates)))
        )
        target: pd.DataFrame = to_frame(
            pd.concat(list(self._split_data(frame=self.target, splitting_dates=splitting_dates)))
        )

        check_len_eq(inputs, target)

        return inputs, target

    def _split_data(
        self, frame: pd.DataFrame, splitting_dates: t.Tuple[t.Optional[float], t.Optional[float]]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        for group in frame.groupby(by=self.group_key):
            indices: np.ndarray = np.arange(len(group[-1].index))
            index_selection: t.List[bool]
            if splitting_dates[0] is None:
                # test dataset
                index_selection = np.quantile(a=indices, q=splitting_dates[-1]) <= indices
            elif splitting_dates[-1] is None:
                # training dataset
                index_selection = indices < np.quantile(a=indices, q=splitting_dates[0])
            else:
                # validation dataset
                index_selection = (np.quantile(a=indices, q=splitting_dates[0]) <= indices) & (
                    indices < np.quantile(a=indices, q=splitting_dates[-1])
                )
            yield group[-1].loc[index_selection, :]

    @staticmethod
    def _get_group(
        group_key: str, group: str, inputs: pd.DataFrame, outputs: pd.DataFrame
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        inputs = inputs.query(f"{group_key} == '{group}'")
        outputs = outputs.query(f"{group_key} == '{group}'")
        return inputs, outputs

    def get_group_set(self, inputs: pd.DataFrame, outputs: pd.DataFrame, group_key: str, group: str):
        return self._get_group(group_key=group_key, group=group, inputs=inputs, outputs=outputs)

    @property
    def training(self) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        return self._dataset(splitting_dates=(self.splitting_proportions[0], None))

    @property
    def validation(self) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        return self._dataset(splitting_dates=(self.splitting_proportions[0], (1 - self.splitting_proportions[-1])))

    @property
    def testing(self) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        return self._dataset(splitting_dates=(None, (1 - self.splitting_proportions[-1])))


@typechecked
def to_frame(frame_or_series: t.Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(frame_or_series, pd.Series):
        return frame_or_series.to_frame()
    return frame_or_series


@typechecked
def check_len_eq(inputs: t.Union[pd.Series, pd.DataFrame], target: t.Union[pd.Series, pd.DataFrame]) -> None:
    if len(inputs) != len(target):
        raise ValueError(f"Len of inputs {len(inputs)} and len of target {len(target)} differ!")


@typechecked
def check_indices(inputs: t.Union[pd.Series, pd.DataFrame], target: t.Union[pd.Series, pd.DataFrame]) -> None:
    if not np.all(np.equal(inputs.index, target.index)):
        raise ValueError("Indices of inputs and target do not match!")


@typechecked
def check_shift_dates(inputs: t.Union[pd.Series, pd.DataFrame], target: t.Union[pd.Series, pd.DataFrame]) -> None:
    if not np.all(inputs.index.get_level_values("date") < target.index.get_level_values("date")):
        raise ValueError("Target is not shifted forward.")


@torch.no_grad()
def init_parameters(parameters: t.Iterator[Parameter], initializer_name: str = "xavier_normal_") -> None:
    with contextlib.suppress(AttributeError):
        for parameter in parameters:
            init_parameter(parameter, initializer_name)


@torch.no_grad()
def init_parameter(
    parameter: t.Union[Parameter, torch.Tensor], initializer_name: str = "xavier_normal_", return_tensor: bool = False
) -> t.Optional[torch.Tensor]:
    try:
        getattr(nn.init, initializer_name)(parameter)
    except ValueError:
        nn.init.normal_(parameter)
    if return_tensor:
        return parameter


def unity(array: np.ndarray) -> np.ndarray:
    return array


softmax: t.Callable[[np.ndarray], np.ndarray] = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def scaled_dot_attention(input: np.ndarray) -> np.ndarray:
    query: np.ndarray = input.view()
    key: np.ndarray = input.view()
    values: np.ndarray = input.view()
    return softmax(query @ key.T / input.shape[-1]) @ values


def are_grads_ok(parameters: t.Iterator[Parameter], verbose: bool = True) -> bool:
    is_okay: bool = True
    for parameter in parameters:
        try:
            grad = parameter.grad
            if grad is None or torch.all(grad == 0.0).item():
                if verbose:
                    logging.warning(
                        "\n".join([f"Please recheck the gradients of parameter={parameter}.", f"grad={grad}"])
                    )
                is_okay = False
        except AttributeError:
            continue
    return is_okay


class IncrementalSecondMoment(Metric):
    """
    Calculate the first and second moments (mean and variance)

    Info: The base class 'Metric' is just used for its design patter, but not as a metric.
    """

    size: t.Tuple[int, int]
    full_state_update: bool = False

    def __init__(self, size: t.Tuple[int, int], **kwargs: t.Any) -> None:
        super().__init__(**kwargs)
        self.size = size
        self.reset()

    def update(self, batch: torch.Tensor) -> None:
        """
        A Jacobian usually has the size: (batch_size, sequence_len, input_size) => dim=3
        A Hessian usually has the size: (batch_size, sequence_len, input_size, input_size) => dim = 4
        """
        # check size
        if batch.size(-1) != self.size[-1]:
            raise ValueError("batch.size(-1) != self.size[-1] or batch_squared.size(-1) != self.size[-1]")

        # update states
        einsum: str = "...j->j" if batch.size(-2) != batch.size(-1) and batch.dim() < 4 else "...ij->ij"
        self._sum_of_squared_batches += torch.einsum(einsum, torch.square(batch))
        self._sum_of_batches += torch.einsum(einsum, batch)
        self._len_of_batches += batch.size(0) if batch.dim() == 2 else batch.size(0) * batch.size(1)

    def compute(self) -> torch.Tensor:
        """Return the variance"""
        return (
            1
            / self._len_of_batches
            * (self._sum_of_squared_batches - 1 / self._len_of_batches * self._sum_of_batches**2)
        ).cpu()

    def reset(self) -> None:
        self.add_state("_sum_of_squared_batches", default=torch.zeros(self.size), dist_reduce_fx="sum")
        self.add_state("_sum_of_batches", default=torch.zeros(self.size), dist_reduce_fx="sum")
        self.add_state("_len_of_batches", default=torch.zeros(1), dist_reduce_fx="sum")

    @property
    def sign_of_sum_of_batches(self) -> torch.Tensor:
        """Return the signs"""
        return torch.sign(self._sum_of_batches).cpu()

    @property
    def sum_of_batches(self) -> torch.Tensor:
        return self._sum_of_batches.cpu()

    @property
    def standard_deviation(self) -> torch.Tensor:
        return torch.sqrt(self.compute()).cpu()

    @property
    def mean(self) -> torch.Tensor:
        """Unadjusted mean"""
        return (1 / self._len_of_batches * self._sum_of_batches).cpu()

    @property
    def positive_valued_mean(self) -> torch.Tensor:
        """Useful because negative and positive values do not cancel each other out as the squared sum is being used"""
        return (1 / self._len_of_batches * torch.sqrt(self._sum_of_squared_batches)).cpu()

    @property
    def signed_mean(self) -> torch.Tensor:
        """Adjusted mean with the squared sum as basis"""
        return (self.sign_of_sum_of_batches * self.positive_valued_mean).cpu()

    @property
    def len_of_batches(self) -> torch.Tensor:
        return self._len_of_batches.cpu()


def split_in_metric_and_type(x: pd.Series) -> pd.DataFrame:
    metric_names: t.List[str] = []
    type_names: t.List[str] = []
    for name in x:
        groups: t.Dict[str, str] = re.search(r"^(?P<type>[a-zA-Z]*)_{1}(?P<metric>.*)$", name).groupdict()
        metric_name: str = groups["metric"]
        metric_names.append(metric_name)
        type: str = groups["type"]
        type_names.append(type)
    return pd.DataFrame({"type": type_names, "metric": metric_names}, index=x.index)


def metric_to_str(metrics: t.Dict[str, torch.Tensor]) -> str:
    return ", ".join(f"{name}={float_to_str(value.item())}" for name, value in metrics.items())


def history_to_plot(
    history: t.Dict[str, t.List[float]],
    scales: t.Optional[str] = None,
    size: t.Optional[t.Tuple[int, int]] = None,
    name: str = "history",
    path: t.Optional[str] = None,
    labs_args: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Optional[PlotSpec]:
    if labs_args is None:
        labs_args = {}
    frame: pd.DataFrame = pd.DataFrame(history)
    frame_melt: pd.DataFrame = frame.melt(ignore_index=False).reset_index()
    frame_melt = frame_melt.merge(
        split_in_metric_and_type(frame_melt["variable"]), left_index=True, right_index=True
    ).drop(columns="variable")

    plot: PlotSpec = (
        (ggplot(frame_melt, aes(x="index", y="value", group="type", color="type")) + geom_line())
        + geom_point()
        + facet_grid(y="metric", scales=scales)
        + labs(**labs_args)
        + GGPLOT_THEME()
    )
    if size is not None:
        plot = plot + ggsize(*size)

    if path is None:
        return plot

    ggsave(plot, os.path.join(path, f"{name}.svg"))
    ggsave(plot, os.path.join(path, f"{name}.html"))


def split_name_of_parameter(name: str) -> t.Dict[str, str]:
    """Returns a dict with the 'object_name', 'module_name' and 'parameter_name'"""
    object_name: str = "object_name"
    module_name: str = "module_name"
    parameter_name: str = "parameter_name"
    reg_ex: re.Pattern = re.compile(f"^(?P<{object_name}>.*)\.(?P<{module_name}>.*)\.(?P<{parameter_name}>.*)$")
    return re.search(reg_ex, name).groupdict()
