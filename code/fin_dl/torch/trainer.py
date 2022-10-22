import numpy as np
import pandas as pd
import torch
import typing as t
from collections import defaultdict
from termcolor import colored
from tqdm import tqdm
from typeguard import typechecked

from fin_dl.torch.callbacks import (
    BreakLoopException,
    EarlyStopping,
    EarlyTrialPruning,
    ParameterCollector,
    ReduceLROnPlateauWrapper,
)
from fin_dl.torch.dataset import StockDataloader
from fin_dl.torch.model import _Constants, Model
from fin_dl.torch.utils import metric_to_str


class Trainer(_Constants):
    model: Model
    dataloader: StockDataloader
    leave_progressbar: bool
    bar_color: str

    @typechecked
    def __init__(
        self,
        model: Model,
        dataloader: StockDataloader,
        leave_progressbar: bool = True,
        bar_color: t.Optional[str] = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.leave_progressbar = leave_progressbar
        self.bar_color = bar_color

    @typechecked
    def fit(
        self,
        max_epochs: int,
        early_stopping: t.Optional[EarlyStopping] = None,
        reduce_on_plateau: t.Optional[ReduceLROnPlateauWrapper] = None,
        parameter_collector: t.Optional[ParameterCollector] = None,
        early_pruning: t.Optional[EarlyTrialPruning] = None,
    ) -> t.Optional[t.Dict[str, t.List[float]]]:
        return self._fit(
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            reduce_on_plateau=reduce_on_plateau,
            parameter_collector=parameter_collector,
            early_pruning=early_pruning,
        )

    def _fit(
        self,
        max_epochs: int,
        early_stopping: t.Optional[EarlyStopping],
        early_pruning: t.Optional[EarlyTrialPruning],
        reduce_on_plateau: t.Optional[ReduceLROnPlateauWrapper],
        parameter_collector: t.Optional[ParameterCollector],
    ) -> t.Optional[t.Dict[str, t.List[float]]]:
        # init history
        history: defaultdict = defaultdict(list)

        if early_stopping is not None:
            # reset early stopping
            early_stopping.reset()
        try:
            # run epochs
            for epoch in np.arange(1, max_epochs + 1):

                # training
                self.model.neuronal_net.train()
                for batch in (
                    prog_bar_train := tqdm(
                        self.dataloader.training,
                        desc=f"Epoch {epoch}/{max_epochs} | Training",
                        leave=self.leave_progressbar,
                        colour=self.bar_color,
                    )
                ):
                    metrics: t.Dict[str, torch.Tensor] = self.model.training_step(batch)
                    prog_bar_train.set_postfix_str(metric_to_str(metrics))

                # append training metrics to history
                for metric_name, metric in self.model.training_epoch_end().items():
                    history[metric_name].append(metric.item())

                if parameter_collector is not None:
                    # collect parameters
                    parameter_collector.update(self.model.neuronal_net.named_parameters(), epoch=epoch)

                # validation
                self.model.neuronal_net.eval()
                for batch in (
                    prog_bar_val := tqdm(
                        self.dataloader.validation,
                        desc=f"Epoch {epoch}/{max_epochs} | Validation",
                        leave=self.leave_progressbar,
                        colour=self.bar_color,
                    )
                ):
                    metrics: t.Dict[str, torch.Tensor] = self.model.validation_step(batch)
                    prog_bar_val.set_postfix_str(metric_to_str(metrics))

                # append validation metrics to history
                for metric_name, metric in self.model.validation_epoch_end().items():
                    history[metric_name].append(metric.item())

                for early_stopping_callbacks in [early_stopping, early_pruning]:
                    if early_stopping_callbacks is not None:
                        # check for early stopping
                        early_stopping_callbacks.update_and_check(float(history[early_stopping_callbacks.monitor][-1]))

                if reduce_on_plateau is not None:
                    # update learning rate
                    reduce_on_plateau.step(history[reduce_on_plateau.monitor][-1])
        except BreakLoopException as e:
            print(colored(str(e), "red"))

        return history

    def evaluate(self) -> t.Dict[str, float]:
        return self._evaluate()

    def _evaluate(self) -> t.Dict[str, float]:
        # init history
        history: defaultdict = defaultdict(list)

        # evaluation
        self.model.neuronal_net.eval()
        for batch in (
            prog_bar := tqdm(
                self.dataloader.testing, desc="Evaluation", leave=self.leave_progressbar, colour=self.bar_color
            )
        ):
            metrics: t.Dict[str, torch.Tensor] = self.model.evaluation_step(batch)
            prog_bar.set_postfix_str(metric_to_str(metrics))

        # append evaluation metrics to history
        for metric_name, metric in self.model.validation_epoch_end().items():
            history[metric_name].append(metric.item())

        return history

    def predict(self, return_metrics: bool = True) -> t.Union[t.Tuple[pd.DataFrame, defaultdict], pd.DataFrame]:
        return self._predict(return_metrics)

    def _predict(
        self, return_metrics: bool = True
    ) -> t.Union[t.Tuple[pd.DataFrame, pd.DataFrame, defaultdict], pd.DataFrame]:

        # init container
        predictions: t.List[np.ndarray] = []
        targets: t.List[np.ndarray] = []

        # init metrics
        metrics_total: defaultdict = defaultdict(list)
        cum_batch_metrics: defaultdict = defaultdict(list)

        # evaluation
        self.model.neuronal_net.eval()
        for batch in (
            prog_bar := tqdm(
                self.dataloader.predicting, desc="Prediction", leave=self.leave_progressbar, colour=self.bar_color
            )
        ):
            # get predictions and targets
            pred_batch: torch.Tensor
            targets_batch: torch.Tensor
            metrics_step: t.Dict[str, torch.Tensor]
            pred_batch, targets_batch, metrics_step = self.model.prediction_step(batch)

            # append to lists
            predictions.append(pred_batch.cpu().detach().squeeze().numpy().reshape(-1, 1))
            targets.append(targets_batch.cpu().detach().squeeze().numpy().reshape(-1, 1))

            # update prog_bar
            prog_bar.set_postfix_str(metric_to_str(metrics_step))

            # append batch metrics to cum_batch_metrics
            for metric_name, metric in metrics_step.items():
                cum_batch_metrics[metric_name].append(float(metric.item()))

        # append prediction metrics to metrics
        for metric_name, metric in self.model.prediction_epoch_end().items():
            metrics_total[metric_name].append(metric.item())

        frame: pd.DataFrame = pd.DataFrame(
            {
                self.LABEL_PREDICTIONS: np.concatenate(predictions).flatten(),
                self.LABEL_TARGET: np.concatenate(targets).flatten(),
            }
        )

        return frame, pd.DataFrame(cum_batch_metrics), metrics_total if return_metrics else frame
