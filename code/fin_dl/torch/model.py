import optuna
import torch
import torchsnooper
import typing as t
from collections import defaultdict
from optuna.trial import FrozenTrial
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from typeguard import typechecked

from fin_dl.torch.anns import CustomFNN, CustomLSTM
from fin_dl.torch.callbacks import BrokenGradientsException, NaNInfException
from fin_dl.torch.metrics import NaiveR2, PearsonCorr, R2
from fin_dl.torch.optuna_config import Optimizer, OptunaChoices, OptunaSuggestOptimizer


class _Constants:
    R2_LABEL_TEMPLATE: str = "{prefix}_r2"
    NAIVE_R2_LABEL_TEMPLATE: str = "{prefix}_naive_r2"
    PEARSON_CORR_LABEL_TEMPLATE: str = "{prefix}_pearson_corr"

    PREFIX_TRAIN: str = "train"
    PREFIX_VAL: str = "val"
    PREFIX_TEST: str = "test"
    PREFIX_PRED: str = "pred"

    LABEL_TARGET: str = "targets"
    LABEL_LOSS: str = "loss"
    LABEL_PREDICTIONS: str = "predictions"


class Model(_Constants):
    neuronal_net: t.Union[CustomLSTM, CustomFNN]
    trial: t.Union[optuna.Trial, FrozenTrial]

    r2: t.Dict[str, R2] = {}
    naive_r2: t.Dict[str, NaiveR2] = {}
    pearson_corr: t.Dict[str, PearsonCorr] = {}

    history: defaultdict
    log_container: t.Dict[str, defaultdict]

    loss: torch.nn.modules.loss
    loss_label_template: str

    optimizer: torch.optim.Optimizer
    device: torch.device

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int,
        sequence_len: int,
        trial: t.Union[optuna.Trial, FrozenTrial],
        ann_type: t.Optional[str] = None,
        final_activation_name: t.Optional[str] = None,
        optuna_choices: OptunaChoices = OptunaChoices(),
        return_sequence: bool = False,
        verbose: bool = True,
        device: t.Optional[t.Union[torch.device, str]] = None,
    ) -> None:
        super().__init__()

        # init attributes
        self.trial = trial
        self.history = defaultdict(list)
        self.device = device

        # draw loss name and set attributes
        loss_name: str = trial.suggest_categorical(name="loss_name", choices=optuna_choices.LOSS_NAMES)
        self.loss = getattr(nn, loss_name)()
        self.loss_label_template = "{prefix}_" + loss_name.lower()

        # init metrics
        for metric_attribute, metric_object in zip(
            [self.r2, self.naive_r2, self.pearson_corr], [R2, NaiveR2, PearsonCorr]
        ):
            metric_attribute.update(
                {
                    metric_prefix: metric_object().to(self.device)
                    for metric_prefix in [self.PREFIX_TRAIN, self.PREFIX_VAL, self.PREFIX_TEST, self.PREFIX_PRED]
                }
            )

        if ann_type is None:
            # draw ann type
            ann_type = trial.suggest_categorical(name="ann_type", choices=optuna_choices.ANN_TYPES)

        # init neuronal net
        if ann_type == "fnn":
            self.neuronal_net = CustomFNN(
                input_size=input_size,
                output_size=output_size,
                sequence_len=sequence_len,
                trial=trial,
                optuna_choices=optuna_choices,
                final_activation_name=final_activation_name,
                verbose=verbose,
            )
        elif ann_type == "lstm":
            self.neuronal_net = CustomLSTM(
                input_size=input_size,
                output_size=output_size,
                trial=trial,
                optuna_choices=optuna_choices,
                final_activation_name=final_activation_name,
                sequence_len=sequence_len,
                return_sequence=return_sequence,
                verbose=verbose,
            )

        # init parameters (xavier)
        self.neuronal_net.init_parameters()

        # to self.device
        self.neuronal_net = self.neuronal_net.to(self.device)

        # draw optimizer and set attribute
        drawn_optimizer: Optimizer = OptunaSuggestOptimizer().get_optimizer(trial=self.trial)
        self.optimizer = getattr(torch.optim, drawn_optimizer.name)(
            self.neuronal_net.parameters(), **drawn_optimizer.args
        )

        if verbose:
            print("\n".join([f"Type of ANN: {ann_type}", f"Optimizer: {drawn_optimizer.name}", f"Loss: {loss_name}"]))

    # @torchsnooper.snoop()
    def _step(self, batch: DataLoader) -> t.Dict[str, torch.Tensor]:
        inputs: torch.Tensor
        target: torch.Tensor
        inputs, target = batch

        # get predictions
        predictions: torch.Tensor = self.neuronal_net.forward(inputs)

        # check for nans and infs
        if torch.any(torch.isnan(predictions)) or torch.any(torch.isinf(predictions)):
            raise NaNInfException("Predictions contain nans/infs.")

        # remove useless dimension
        if predictions.dim() == 3 and predictions.size(-1) == 1:
            predictions.squeeze_(-1)

        # calc loss
        loss: torch.Tensor
        if isinstance(self.loss, torch.nn.modules.loss.GaussianNLLLoss):
            loss: torch.Tensor = self.loss(
                predictions, target, torch.ones(target.size(0), 1, requires_grad=True, device=self.device)
            )
        else:
            loss: torch.Tensor = self.loss(predictions, target)

        # return loss, pred, target
        return {self.LABEL_LOSS: loss.to(self.device), self.LABEL_PREDICTIONS: predictions, self.LABEL_TARGET: target}

    def _epoch_end(self, prefix: str) -> t.Dict[str, torch.Tensor]:
        self._log_metrics_epoch_end(prefix)
        metrics: t.Dict[str, torch.Tensor] = self._get_metrics(prefix)
        self._reset_metrics(prefix)
        return metrics

    def _update_metrics_step_end(self, step_output: t.Dict[str, torch.Tensor], prefix: str) -> None:
        for metric in [self.r2[prefix], self.naive_r2[prefix], self.pearson_corr[prefix]]:
            self._update_metric(step_output[self.LABEL_PREDICTIONS], step_output[self.LABEL_TARGET], metric=metric)

    @staticmethod
    def _update_metric(pred: torch.Tensor, target: torch.Tensor, metric: Metric) -> None:
        try:
            metric.update(pred, target)
        except RuntimeError as e:
            raise RuntimeError(
                "\n".join(
                    [str(e), f"pred: {pred}", f"target: {target}", f"metric={metric}", f"metric.device={metric.device}"]
                )
            ) from e

    # noinspection StrFormat
    def _get_metrics(self, prefix: str) -> t.Dict[str, torch.Tensor]:
        return {
            self.R2_LABEL_TEMPLATE.format(prefix=prefix): self.r2[prefix].compute(),
            self.NAIVE_R2_LABEL_TEMPLATE.format(prefix=prefix): self.naive_r2[prefix].compute(),
            self.PEARSON_CORR_LABEL_TEMPLATE.format(prefix=prefix): self.pearson_corr[prefix].compute(),
        }

    def _log_metrics(self, prefix: str, **kwargs) -> None:
        metric_results: t.Dict[str, torch.Tensor] = self._get_metrics(prefix)
        for metric_name, metric_value in metric_results.items():
            self.history[metric_name].append(metric_value.item())

    def _log_metrics_epoch_end(self, prefix: str) -> None:
        # log the metrics for the complete epoch
        self._log_metrics(prefix)

    def _reset_metrics(self, prefix: str) -> None:
        # reset each metric after the epoch has ended
        for metric in [self.r2, self.naive_r2, self.pearson_corr]:
            metric[prefix].reset()
            metric[prefix] = metric[prefix].to(self.device)

    def training_step(self, batch: DataLoader) -> t.Dict[str, torch.Tensor]:
        output: t.Dict[str, torch.Tensor] = self._step(batch=batch)

        # back-propagate and update optimizer
        self.optimizer.zero_grad()
        output[self.LABEL_LOSS].backward()
        # clipping gradients
        # torch.nn.utils.clip_grad_norm_(self.neuronal_net.parameters(), 1)
        self.optimizer.step()

        return self.training_step_end(output)

    def training_step_end(self, step_output: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        self._update_metrics_step_end(step_output, prefix=self.PREFIX_TRAIN)
        return self._get_metrics(self.PREFIX_TRAIN)

    def training_epoch_end(self) -> t.Dict[str, torch.Tensor]:
        # check if gradients are okay
        if not self.neuronal_net.are_grads_ok(verbose=False):
            raise BrokenGradientsException("Gradients are broken.")

        return self._epoch_end(self.PREFIX_TRAIN)

    def validation_step(self, batch: DataLoader) -> t.Dict[str, torch.Tensor]:
        output: t.Dict[str, torch.Tensor] = self._step(batch=batch)
        return self.validation_step_end(output)

    def validation_step_end(self, step_output: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        self._update_metrics_step_end(step_output, prefix=self.PREFIX_VAL)
        return self._get_metrics(self.PREFIX_VAL)

    def validation_epoch_end(self) -> t.Dict[str, torch.Tensor]:
        return self._epoch_end(self.PREFIX_VAL)

    def evaluation_step(self, batch: DataLoader) -> t.Dict[str, torch.Tensor]:
        output: t.Dict[str, torch.Tensor] = self._step(batch=batch)
        return self.evaluation_step_end(output)

    def evaluation_step_end(self, step_output: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        self._update_metrics_step_end(step_output, prefix=self.PREFIX_TEST)
        return self._get_metrics(self.PREFIX_TEST)

    def evaluation_epoch_end(self) -> t.Dict[str, torch.Tensor]:
        return self._epoch_end(self.PREFIX_TEST)

    def prediction_step(self, batch: DataLoader) -> t.Tuple[torch.Tensor, torch.Tensor, t.Dict[str, torch.Tensor]]:
        output: t.Dict[str, torch.Tensor] = self._step(batch=batch)
        return self.prediction_step_end(output)

    def prediction_step_end(
        self, step_output: t.Dict[str, torch.Tensor]
    ) -> t.Tuple[torch.Tensor, torch.Tensor, t.Dict[str, torch.Tensor]]:
        self._update_metrics_step_end(step_output, prefix=self.PREFIX_PRED)
        return step_output[self.LABEL_PREDICTIONS], step_output[self.LABEL_TARGET], self._get_metrics(self.PREFIX_PRED)

    def prediction_epoch_end(self) -> t.Dict[str, torch.Tensor]:
        return self._epoch_end(self.PREFIX_PRED)

    def __repr__(self) -> str:
        return str(self.neuronal_net)
