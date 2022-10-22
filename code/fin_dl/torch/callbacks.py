import typing as t

import numpy as np
import pandas as pd
from attr import define, field
from lets_plot.plot.core import aes, PlotSpec
from termcolor import cprint
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typeguard import typechecked

from fin_dl.plotting.plots import facet_violin_plot
from fin_dl.torch.utils import split_name_of_parameter
from fin_dl.utilities import FileStructure, float_to_str


@define
class EarlyStopping:
    monitor: str
    direction: str
    patience: int
    min_delta: float

    _logged_value: t.List[float] = []
    _comparison_callable: t.Callable = field(init=False)

    @typechecked
    def __init__(self, monitor: str, patience: int, direction: str, min_delta: float) -> None:
        self.__attrs_init__(monitor=monitor, patience=patience, direction=direction, min_delta=min_delta)

    @_comparison_callable.default
    def _set_comparison_callable(self) -> t.Callable:
        if self.direction == "min":
            return np.less
        elif self.direction == "max":
            return np.greater

    def reset(self) -> None:
        self._logged_value = []

    @typechecked
    def update_and_check(self, new_value: float) -> None:
        self._logged_value.append(new_value)
        self._check()

    def _check(self) -> None:
        if len(self._logged_value) > self.patience:
            recent: float = self._logged_value[-1]
            older: float = self._logged_value[-(self.patience + 1)]
            if np.isclose(recent, older, atol=self.min_delta) or not self._comparison_callable(recent, older):
                raise EarlyStoppingException("Triggered early stopping.")


class EarlyTrialPruning(EarlyStopping):
    hard_prune_is_close_factor: t.Optional[float]
    hard_prune_min_threshold: float
    baseline: t.Optional[float] = None

    @typechecked
    def __init__(
        self,
        monitor: str,
        patience: int,
        direction: str,
        min_delta: float,
        hard_prune_is_close_factor: t.Optional[float] = None,
        hard_prune_min_threshold: float = -1.0,
    ) -> None:
        super().__init__(
            monitor=monitor,
            patience=patience,
            direction=direction,
            min_delta=min_delta,
        )
        self.hard_prune_is_close_factor = hard_prune_is_close_factor
        self.hard_prune_min_threshold = hard_prune_min_threshold

    @typechecked
    def update_and_check(self, new_value: float) -> None:
        """Used within the trainer fit()"""
        self._logged_value.append(new_value)
        self._normal_check()
        if self.hard_prune_is_close_factor is not None:
            self._hard_check()

    def _normal_check(self) -> None:
        new_value: float = self._logged_value[-1]
        if (
            self.baseline is not None
            and len(self._logged_value) > self.patience
            and (
                np.isclose(new_value, self.baseline, atol=self.min_delta)
                or not self._comparison_callable(new_value, self.baseline)
            )
        ):
            raise EarlyTrialPruningException("Tuning trial pruned early.")

    def _hard_check(self) -> None:
        """
        - If the new value is not close to 'self.baseline * self.hard_prune_is_close_factor' => raise exception.
        - Ignores the patience value.
        """
        new_value: float = self._logged_value[-1]
        if self.baseline is not None and (
            not np.isclose(new_value, self.baseline, rtol=self.hard_prune_is_close_factor)
            and not self._comparison_callable(new_value, self.baseline)
            and not self._comparison_callable(new_value, self.hard_prune_min_threshold)
        ):
            raise EarlyTrialPruningException(
                f"Tuning trial pruned early (baseline={float_to_str(self.baseline)} and this epoch's value="
                f"{float_to_str(new_value)})."
            )

    @typechecked
    def update_baseline(self, new_value: float) -> None:
        """Used within the tuner tune()"""
        if ((self.baseline is None or np.isnan(self.baseline)) and not np.isnan(new_value)) or (
            not np.isclose(new_value, self.baseline, atol=self.min_delta)
            and self._comparison_callable(new_value, self.baseline)
        ):
            self.baseline = new_value
            cprint(
                f"Set new baseline for early pruning callback: {self.monitor}={float_to_str(self.baseline)}", "green"
            )
        else:
            print(f"Best value is still: {self.monitor}={float_to_str(self.baseline)}")

    def reset(self) -> None:
        super().reset()

    def reset_baseline(self) -> None:
        self.baseline = None


class ReduceLROnPlateauWrapper:
    monitor: str
    reduce_on_plateau: ReduceLROnPlateau
    base_kwargs: t.Dict[str, t.Any]

    def __init__(
        self,
        monitor: str,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        self.monitor = monitor
        self.base_kwargs = dict(
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )

    def set_optimizer(
        self,
        optimizer: Optimizer,
    ) -> None:
        self.reduce_on_plateau = ReduceLROnPlateau(optimizer=optimizer, **self.base_kwargs)

    def step(self, metrics: t.Any) -> None:
        self.reduce_on_plateau.step(metrics=metrics)


@define
class ParameterCollector:
    create_plots_after_epoch: bool
    file_structure: FileStructure
    name: str
    verbose: bool

    container_weight: t.List[pd.DataFrame] = field(init=False)
    container_bias: t.List[pd.DataFrame] = field(init=False)
    container_weight_grads: t.List[pd.DataFrame] = field(init=False)
    container_bias_grads: t.List[pd.DataFrame] = field(init=False)

    def __init__(
        self, file_structure: FileStructure, name: str, create_plots_after_epoch: bool = True, verbose: bool = True
    ) -> None:
        """Setting 'create_plots_after_epoch' to False can be 'dangerous', as this accumulates all parameters per
        epoch step, leading to high memory usage if the model has many parameters (i. e., very deep)"""
        self.__attrs_init__(
            file_structure=file_structure, name=name, create_plots_after_epoch=create_plots_after_epoch, verbose=verbose
        )

    def __attrs_post_init__(self) -> None:
        self.reset_states()

    def reset_states(self) -> None:
        self.container_weight = []
        self.container_bias = []
        self.container_weight_grads = []
        self.container_bias_grads = []

    def print(self, text: str, color: str = "green", **kwargs) -> None:
        if self.verbose:
            cprint(text, color, **kwargs)

    def update(self, named_parameters: t.Iterator[t.Tuple[str, Parameter]], epoch: int) -> None:
        for name, parameter in named_parameters:
            # split name into its parts
            parts: t.Dict[str, str] = split_name_of_parameter(name)

            # convert to pandas dataframe
            # parameters
            parameter_as_array: np.ndarray = parameter.detach().cpu().numpy().flatten().reshape(1, -1)
            parameter_as_frame: pd.DataFrame = (
                pd.DataFrame(
                    parameter_as_array,
                    columns=np.repeat(epoch, parameter_as_array.shape[-1]),
                    index=[f"{parts['module_name']}.{parts['parameter_name']}"],
                )
                .T.melt(ignore_index=False)
                .reset_index()
            )

            # gradients
            grads_as_array: np.ndarray = parameter.grad.detach().cpu().numpy().flatten().reshape(1, -1)
            grads_as_frame: pd.DataFrame = (
                pd.DataFrame(
                    grads_as_array,
                    columns=np.repeat(epoch, parameter_as_array.shape[-1]),
                    index=[f"{parts['module_name']}.{parts['parameter_name']}"],
                )
                .T.melt(ignore_index=False)
                .reset_index()
            )

            # append to lists
            if "weight" in parts["parameter_name"]:
                self.container_weight.append(parameter_as_frame)
                self.container_weight_grads.append(grads_as_frame)
            elif "bias" in parts["parameter_name"]:
                self.container_bias.append(parameter_as_frame)
                self.container_bias_grads.append(grads_as_frame)

        if self.create_plots_after_epoch:
            self.plot_all(epoch=epoch)
            self.reset_states()

    @staticmethod
    def _standard_plot(
        frame: pd.DataFrame,
        kernel: str = "epanechikov",
        name: t.Optional[str] = None,
        path: t.Optional[str] = None,
        subtitle: t.Optional[str] = None,
    ) -> t.Optional[PlotSpec]:
        return facet_violin_plot(
            frame,
            aes_args=dict(x="variable", y="value"),
            facet_args=dict(x="index"),
            violin_args=dict(mapping=aes(fill="variable"), alpha=0.5, kernel=kernel),
            path=path,
            name=name,
            labs_args=dict(title="Parameter Distribution", subtitle=f"{subtitle} (kernel='{kernel}')", x="", y=""),
        )

    def plot_parameter_weight(
        self, path: t.Optional[str] = None, name: t.Optional[str] = None, kernel_name: t.Optional[str] = "epanechikov"
    ) -> t.Optional[PlotSpec]:
        return self._standard_plot(
            frame=self.parameter_weight, path=path, name=name, subtitle="Weight", kernel=kernel_name
        )

    def plot_parameter_weight_grads(
        self, path: t.Optional[str] = None, name: t.Optional[str] = None, kernel_name: t.Optional[str] = "epanechikov"
    ) -> t.Optional[PlotSpec]:
        return self._standard_plot(
            frame=self.parameter_weight_grads, path=path, name=name, subtitle="Weight Gradients", kernel=kernel_name
        )

    def plot_parameter_bias(
        self, path: t.Optional[str] = None, name: t.Optional[str] = None, kernel_name: t.Optional[str] = "epanechikov"
    ) -> t.Optional[PlotSpec]:
        return self._standard_plot(frame=self.parameter_bias, path=path, name=name, subtitle="Bias", kernel=kernel_name)

    def plot_parameter_bias_grads(
        self, path: t.Optional[str] = None, name: t.Optional[str] = None, kernel_name: t.Optional[str] = "epanechikov"
    ) -> t.Optional[PlotSpec]:
        return self._standard_plot(
            frame=self.parameter_bias_grads, path=path, name=name, subtitle="Bias Gradients", kernel=kernel_name
        )

    def plot_all(self, epoch: int) -> None:
        for kernel_name in ["gaussian", "epanechikov"]:
            self.plot_parameter_weight(
                name=f"{self.name}_epoch={epoch}_kernel={kernel_name}",
                path=self.file_structure.best_model_parameters_weight,
            )
            self.print(f"Plotted parameter weights (kernel={kernel_name})")
            self.plot_parameter_weight_grads(
                name=f"{self.name}_epoch={epoch}_kernel={kernel_name}",
                path=self.file_structure.best_model_parameters_weight_grads,
            )
            self.print(f"Plotted gradients of weights (kernel={kernel_name})")
            self.plot_parameter_bias(
                name=f"{self.name}_epoch={epoch}_kernel={kernel_name}",
                path=self.file_structure.best_model_parameters_bias,
            )
            self.print(f"Plotted parameter biases (kernel={kernel_name})")
            self.plot_parameter_bias_grads(
                name=f"{self.name}_epoch={epoch}_kernel={kernel_name}",
                path=self.file_structure.best_model_parameters_bias_grads,
            )
            self.print(f"Plotted gradients of biases (kernel={kernel_name})")

    @property
    def parameter_weight(self) -> pd.DataFrame:
        return pd.concat(self.container_weight)

    @property
    def parameter_weight_grads(self) -> pd.DataFrame:
        return pd.concat(self.container_weight_grads)

    @property
    def parameter_bias(self) -> pd.DataFrame:
        return pd.concat(self.container_bias)

    @property
    def parameter_bias_grads(self) -> pd.DataFrame:
        return pd.concat(self.container_bias_grads)


class BreakLoopException(Exception):
    pass


class StopProcessException(Exception):
    pass


class EarlyStoppingException(BreakLoopException):
    pass


class BrokenGradientsException(StopProcessException):
    pass


class NaNInfException(StopProcessException):
    pass


class EarlyTrialPruningException(StopProcessException, Exception):
    pass
