import typing as t

import numpy as np
import optuna
from attr import define, field


class Optimizer(t.NamedTuple):
    name: str
    args: t.Dict[str, t.Any]


class MinMax(t.NamedTuple):
    min: t.Union[float, int]
    max: t.Union[float, int]


@define
class OptunaSuggestOptimizer:
    NAMES: t.List[str] = ["Adadelta", "Adagrad", "Adam", "Rprop"]
    weight_decay: MinMax = MinMax(min=0.0001, max=0.001)
    choices: t.Dict[str, t.Dict[str, t.Any]] = field()

    @choices.default
    def _set_choices(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        return {
            "Adadelta": {
                "lr": dict(suggest_uniform=dict(name="Adadelta_lr", low=0.01, high=1.0)),
                "rho": dict(suggest_uniform=dict(name="Adadelta_rho", low=0.001, high=1.0)),
                "eps": dict(suggest_uniform=dict(name="Adadelta_eps", low=1e-07, high=1e-05)),
            },
            "Adagrad": {
                "lr": dict(suggest_uniform=dict(name="Adagrad_lr", low=0.001, high=0.1)),
                "lr_decay": dict(suggest_uniform=dict(name="Adagrad_lr_decay", low=0.001, high=0.1)),
                "weight_decay": dict(
                    suggest_uniform=dict(
                        name="Adagrad_weight_decay", low=self.weight_decay.min, high=self.weight_decay.max
                    )
                ),
            },
            "Adam": {
                "lr": dict(suggest_uniform=dict(name="Adam_lr", low=0.0001, high=0.1)),
                "betas": (
                    dict(suggest_uniform=dict(name="Adam_beta1", low=0.85, high=0.95)),
                    dict(suggest_uniform=dict(name="Adam_beta2", low=0.95, high=1.0)),
                ),
                "eps": dict(suggest_uniform=dict(name="Adam_eps", low=1e-9, high=1e-07)),
                "weight_decay": dict(
                    suggest_uniform=dict(
                        name="Adam_weight_decay", low=self.weight_decay.min, high=self.weight_decay.max
                    )
                ),
            },
            "Rprop": {
                "lr": dict(suggest_uniform=dict(name="Rprop_lr", low=0.001, high=0.1)),
                "etas": (
                    dict(suggest_uniform=dict(name="Rprop_eta1", low=0.4, high=0.6)),
                    dict(suggest_uniform=dict(name="Rprop_eta2", low=1.1, high=1.3)),
                ),
                "step_sizes": (
                    dict(suggest_uniform=dict(name="Rprop_step_size1", low=1e-08, high=1e-04)),
                    dict(suggest_uniform=dict(name="Rprop_step_size2", low=1.0, high=50.0)),
                ),
            },
        }

    def get_optimizer(self, trial: optuna.Trial) -> Optimizer:
        optimizer_name: str = trial.suggest_categorical(name="optimizer_name", choices=self.NAMES)
        arguments: t.Dict[str, t.Any] = self.choices[optimizer_name]
        use_default_optimizer_args: bool = trial.suggest_categorical(
            "use_default_optimizer_args", choices=[True, False]
        )
        print(f"Use default optimizer args: {use_default_optimizer_args}")
        return Optimizer(
            name=optimizer_name,
            args={} if use_default_optimizer_args else self._draw_parameters(trial=trial, arguments=arguments),
        )

    def _draw_parameters(self, trial: optuna.Trial, arguments: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        args: t.Dict[str, t.Any] = {}
        for argument_name, value in arguments.items():
            if isinstance(value, dict):
                args.update({argument_name: self._draw_parameter(trial=trial, suggest_name_with_args=value)})
            elif isinstance(value, tuple):
                dictionary: t.Dict[str, dict]
                container: t.List[t.Any] = [
                    self._draw_parameter(trial=trial, suggest_name_with_args=dictionary) for dictionary in value
                ]

                args.update({argument_name: tuple(container)})
        return args

    @staticmethod
    def _draw_parameter(trial: optuna.Trial, suggest_name_with_args: t.Dict[str, t.Any]) -> t.Any:
        key_value: t.Tuple[str, dict] = suggest_name_with_args.popitem()
        optuna_suggest_name: str = key_value[0]
        optuna_suggest_args: t.Dict[str, t.Any] = key_value[-1]
        return getattr(trial, optuna_suggest_name)(**optuna_suggest_args)


@define
class OptunaChoices:
    exponent_units: MinMax
    exponent_dropout_rates: MinMax
    stack_depth: MinMax

    units: t.Tuple[int] = field(init=False)
    dropout_rates: t.Tuple[float] = field(init=False)

    ACTIVATION_FUNCTION_NAMES: t.List[str] = ["SELU", "Tanh", "CELU", "GELU", "SiLU", "Mish", "Softplus", "ELU"]
    ANN_TYPES: t.List[str] = ["lstm", "fnn"]
    LOSS_NAMES: t.List[str] = ["GaussianNLLLoss", "MSELoss", "HuberLoss"]

    def __init__(
        self,
        exponent_units: MinMax = MinMax(4, 10),
        exponent_dropout_rates: MinMax = MinMax(0, 6),
        stack_depth: MinMax = MinMax(1, 6),
    ):
        self.__attrs_init__(
            exponent_units=exponent_units,
            exponent_dropout_rates=exponent_dropout_rates,
            stack_depth=stack_depth,
        )

    def __attrs_post_init__(self) -> None:
        self.units = tuple(int(2**e) for e in np.arange(self.exponent_units.min, self.exponent_units.max + 1))
        self.dropout_rates = tuple(
            float(2.0**e / 100)
            for e in np.arange(self.exponent_dropout_rates.min, self.exponent_dropout_rates.max + 1)
        )
