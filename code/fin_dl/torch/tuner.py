import os
import typing as t

import numpy as np
import optuna
import pandas as pd
import torch
import wandb
from attr import define, field
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState
from termcolor import colored, cprint
from typeguard import typechecked

from fin_dl import ENTITY
from fin_dl.plotting.plots import facet_line_plot, line_plot
from fin_dl.torch import device
from fin_dl.torch.callbacks import (
    EarlyStopping,
    EarlyTrialPruning,
    EarlyTrialPruningException,
    ParameterCollector,
    ReduceLROnPlateauWrapper,
    StopProcessException,
)
from fin_dl.torch.dataset import StockDataloader
from fin_dl.torch.interpretability import PlotModelInsights
from fin_dl.torch.model import _Constants, Model
from fin_dl.torch.trainer import Trainer
from fin_dl.torch.utils import history_to_plot
from fin_dl.utilities import FileStructure, float_to_str, save_optuna_plot, to_yaml


# noinspection StrFormat
@define
class Tuner(_Constants):
    dataloader: StockDataloader
    file_structure: FileStructure
    name: str
    collect_and_plot_parameters: bool
    wandb_project: str

    trainer: Trainer = field(init=False)
    study: optuna.Study = field(init=False)
    model: Model = field(init=False)

    objective: str = field(init=False)

    best_model_eval_predictions: pd.DataFrame = field(init=False)
    best_model_eval_cum_batch_metrics: pd.DataFrame = field(init=False)
    best_model_eval_pred_metrics: t.Dict[str, t.List[float]] = field(init=False)
    best_model_fitting_metrics: t.Dict[str, t.List[float]] = field(init=False)

    def __init__(
        self,
        dataloader: StockDataloader,
        file_structure: FileStructure,
        name: str,
        collect_and_plot_parameters: bool = True,
        wandb_project: str = "test",
    ) -> None:
        self.__attrs_init__(
            dataloader=dataloader,
            file_structure=file_structure,
            name=name,
            collect_and_plot_parameters=collect_and_plot_parameters,
            wandb_project=wandb_project,
        )

    def _fit(
        self,
        trial: t.Union[optuna.Trial, FrozenTrial],
        max_epochs: int,
        early_stopping: t.Optional[EarlyStopping] = None,
        reduce_on_plateau: t.Optional[ReduceLROnPlateauWrapper] = None,
        parameter_collector: t.Optional[ParameterCollector] = None,
        early_pruning: t.Optional[EarlyTrialPruning] = None,
    ) -> t.Optional[t.Dict[str, t.List[float]]]:
        # fit model
        self.model = Model(
            input_size=self.dataloader.input_size,
            output_size=self.dataloader.output_size,
            sequence_len=self.dataloader.sequence_len,
            trial=trial,
            device=device,
        )

        # init trainer
        self.trainer = Trainer(self.model, dataloader=self.dataloader)

        # reset reduce_on_plateau
        if reduce_on_plateau is not None:
            reduce_on_plateau.set_optimizer(self.model.optimizer)

        # reset early_stopping
        if early_stopping is not None:
            early_stopping.reset()

        # fit model
        return self.trainer.fit(
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            reduce_on_plateau=reduce_on_plateau,
            parameter_collector=parameter_collector,
            early_pruning=early_pruning,
        )

    def _eval_best_model(
        self,
        max_epochs_evaluation: int,
        max_epochs_tuning: int,
        trials: int,
        early_stopping: t.Optional[EarlyStopping],
        reduce_on_plateau: t.Optional[ReduceLROnPlateauWrapper],
    ) -> None:
        # print status
        print(colored("Fitting the best model.", "green"))

        # init parameter_collector
        parameter_collector: t.Optional[ParameterCollector] = (
            ParameterCollector(file_structure=self.file_structure, name=self.name)
            if self.collect_and_plot_parameters
            else None
        )

        # reset early_stopping
        if early_stopping is not None:
            early_stopping.reset()

        # fit model with best hyper-parameters
        self.best_model_fitting_metrics = self._fit(
            trial=self.study.best_trial,
            max_epochs=max_epochs_evaluation,
            early_stopping=early_stopping,
            reduce_on_plateau=reduce_on_plateau,
            parameter_collector=parameter_collector,
        )

        while self.best_model_fitting_metrics is None:
            # if 'fitting_metrics' is None due to an exception like NaNInfException, then completely restart tuning
            print(colored("Tuning is completely restarted.", "red", attrs=["bold"]))
            self._tune(
                max_epochs_tuning=max_epochs_tuning,
                trials=trials,
                early_stopping=early_stopping,
            )
            self.best_model_fitting_metrics = self._fit(
                trial=self.study.best_trial,
                max_epochs=max_epochs_evaluation,
                early_stopping=early_stopping,
                reduce_on_plateau=reduce_on_plateau,
            )

        # evaluate
        (
            self.best_model_eval_predictions,
            self.best_model_eval_cum_batch_metrics,
            self.best_model_eval_pred_metrics,
        ) = self.trainer.predict()

    def post_tuning(
        self,
        calc_hessian_for: t.Optional[t.List[str]] = None,
    ) -> None:
        """Save and plot all results"""
        if calc_hessian_for is None:
            calc_hessian_for = ["Training", "Validation", "Testing"]

        # save best parameters
        to_yaml(
            dictionary=self.study.best_trial.params,
            filename=os.path.join(self.file_structure.tuning_data_best_trial, f"{self.name}.yaml"),
        )

        # save predictions
        self.best_model_eval_predictions.to_csv(
            os.path.join(self.file_structure.predictions_values, f"{self.name}.csv")
        )

        # ...plot
        predictions = self.best_model_eval_predictions.melt(ignore_index=False).reset_index()
        line_plot(
            predictions,
            aes_args=dict(x="index", y="value", color="variable"),
            labs_args=dict(title="Predictions", subtitle=self.name, x="", y=""),
            path=self.file_structure.predictions_values,
            name=self.name,
        )

        # save batch pred_metrics
        cum_batch_metrics = self.best_model_eval_cum_batch_metrics.melt(ignore_index=False).reset_index()
        facet_line_plot(
            cum_batch_metrics,
            aes_args=dict(x="index", y="value"),
            facet_args=dict(y="variable", scales="free"),
            labs_args=dict(title="Batch History", subtitle=self.name, x="Batch No.", y=""),
            path=self.file_structure.predictions_metrics_batch_history,
            name=self.name,
        )

        # save pred_metrics
        pd.DataFrame(self.best_model_eval_pred_metrics).to_csv(
            os.path.join(self.file_structure.predictions_metrics, f"{self.name}.csv")
        )

        # save fitting_metrics as plot and .csv
        history_to_plot(
            self.best_model_fitting_metrics,
            name=self.name,
            path=self.file_structure.fitting_history,
            labs_args=dict(
                title="History",
                subtitle="Fitting the best model",
                caption=self.name,
                x="Epoch",
                y="",
            ),
        )
        pd.DataFrame(self.best_model_fitting_metrics).to_csv(
            os.path.join(self.file_structure.fitting_history, f"{self.name}.csv")
        )

        # save the hyperparameter importance as .pdf
        save_optuna_plot(
            path=self.file_structure.tuning_data_plots_hyperparameter_importance,
            name=f"{self.name}.csv",
            study=self.study,
            plot_fun=optuna.visualization.plot_param_importances,
        )

        # save the slice plot as .pdf
        save_optuna_plot(
            path=self.file_structure.tuning_data_plots_plot_slice,
            name=f"{self.name}.csv",
            study=self.study,
            plot_fun=optuna.visualization.plot_slice,
        )

        # save study info as .csv
        self.study.trials_dataframe().to_csv(
            os.path.join(self.file_structure.tuning_data_trials, f"{self.name}.csv"),
            index=True,
        )

        # save model architecture
        with open(os.path.join(self.file_structure.best_model_architecture, f"{self.name}.txt"), "w") as f:
            print(self.model, file=f)

        # save model state dict
        torch.save(
            self.model.neuronal_net.state_dict(),
            os.path.join(self.file_structure.best_model_state_dict, f"{self.name}.pt"),
        )

        # save model named parameters
        torch.save(
            list(self.model.neuronal_net.named_parameters()),
            os.path.join(self.file_structure.best_model_parameters, f"{self.name}.pt"),
        )

        # save model insights
        PlotModelInsights(
            model=self.model.neuronal_net,
            name=self.name,
            file_structure=self.file_structure,
            datasets=self.dataloader,
            calc_hessian_for=calc_hessian_for,
        )

    @typechecked
    def tune(
        self,
        objective: str,
        max_epochs_evaluation: int,
        max_epochs_tuning: int,
        trials: int,
        early_stopping: t.Optional[EarlyStopping] = None,
        early_pruning: t.Optional[EarlyTrialPruning] = None,
        reduce_on_plateau: t.Optional[ReduceLROnPlateauWrapper] = None,
    ) -> None:
        # set objective
        self.objective = objective

        # tune
        self._tune(
            max_epochs_tuning=max_epochs_tuning,
            trials=trials,
            early_stopping=early_stopping,
            early_pruning=early_pruning,
        )

        # best trial
        best_trial: t.Optional[FrozenTrial] = None
        while best_trial is None:
            try:
                best_trial = self.study.best_trial
            except ValueError as e:
                print(e)
                cprint("Restarting tuning", "red")
                self._tune(
                    max_epochs_tuning=max_epochs_tuning,
                    trials=trials,
                    early_stopping=early_stopping,
                    early_pruning=early_pruning,
                )

        # use the best trial to create, fit and evaluate the model
        self._eval_best_model(
            max_epochs_tuning=max_epochs_tuning,
            trials=trials,
            max_epochs_evaluation=max_epochs_evaluation,
            early_stopping=early_stopping,
            reduce_on_plateau=reduce_on_plateau,
        )

    def _tune(
        self,
        max_epochs_tuning: int,
        trials: int,
        early_stopping: t.Optional[EarlyStopping] = None,
        early_pruning: t.Optional[EarlyTrialPruning] = None,
    ) -> None:

        # init optuna
        self.study = optuna.create_study(
            study_name=self.name,
            sampler=TPESampler(),
            direction="maximize",
        )

        if early_pruning is not None:
            # reset early_pruning
            early_pruning.reset_baseline()

        with wandb.init(
            name=self.name,
            project=self.wandb_project,
            entity=ENTITY,
            group=f"ann tuning - {self.name}",
        ):
            for step in range(trials):
                # set prog_bar description
                print(colored(f"Optuna trial {step + 1}/{trials}", "green"))

                # get trial
                trial: optuna.Trial = self.study.ask()

                if early_pruning is not None:
                    # reset early_pruning container
                    early_pruning.reset()
                try:
                    # fit model and get metrics
                    metrics: t.Optional[t.Dict[str, t.List[float]]] = self._fit(
                        trial=trial,
                        max_epochs=max_epochs_tuning,
                        early_stopping=early_stopping,
                        early_pruning=early_pruning,
                    )
                except StopProcessException as e:
                    if isinstance(e, EarlyTrialPruningException):
                        self.study.tell(trial=trial, state=TrialState.PRUNED)
                    else:
                        self.study.tell(trial=trial, state=TrialState.FAIL)
                    print(
                        "\n".join(
                            [
                                colored(f"Optuna trial {step + 1}/{trials} pruned!", "red", attrs=["bold"]),
                                colored(str(e), "red"),
                            ]
                        )
                    )
                    continue

                # noinspection PyTypeChecker
                max_objective_index: int = np.argmax(metrics[self.objective])

                # get single metrics
                naive_r2: float = metrics[self.NAIVE_R2_LABEL_TEMPLATE.format(prefix=self.PREFIX_VAL)][
                    max_objective_index
                ]
                r2: float = metrics[self.R2_LABEL_TEMPLATE.format(prefix=self.PREFIX_VAL)][max_objective_index]
                pearson_corr: float = metrics[self.PEARSON_CORR_LABEL_TEMPLATE.format(prefix=self.PREFIX_VAL)][
                    max_objective_index
                ]

                # get objective
                objective: float = metrics[self.objective][max_objective_index]

                # update study
                self.study.tell(trial=trial, values=objective)

                # update early pruning baseline
                early_pruning.update_baseline(metrics[early_pruning.monitor][max_objective_index])

                # print objective
                print(
                    colored(
                        f"Optuna trial {step + 1}/{trials} => Objective: {self.objective}="
                        f"{float_to_str(objective)}",
                        "blue",
                    )
                )

                # log to wandb
                wandb.log(dict(**trial.params, pearson_corr=pearson_corr, r2=r2, naive_r2=naive_r2), step=step)
