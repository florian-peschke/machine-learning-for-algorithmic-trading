import os
import typing as t

from attr import define

from fin_dl.rolling_predictions.static import StaticWindows
from fin_dl.torch.callbacks import EarlyStopping, EarlyTrialPruning, ReduceLROnPlateauWrapper
from fin_dl.torch.dataset import StockDataloader
from fin_dl.torch.tuner import Tuner
from fin_dl.utilities import FileStructure


@define
class RollingWindowPrediction:
    dataloader_config: t.Dict[str, t.Any]
    early_stopping: t.Optional[EarlyStopping]
    early_pruning: t.Optional[EarlyTrialPruning]
    file_structure: FileStructure
    max_epochs_evaluation: int
    max_epochs_tuning: int
    reduce_on_plateau: t.Optional[ReduceLROnPlateauWrapper]
    tuning_trials: int
    window_maker: StaticWindows
    wandb_project: str
    calc_hessian_for: t.List[str]

    def __init__(
        self,
        dataloader_config: t.Dict[str, t.Any],
        file_structure: FileStructure,
        max_epochs_evaluation: int,
        max_epochs_tuning: int,
        tuning_trials: int,
        window_maker: StaticWindows,
        reduce_on_plateau: t.Optional[ReduceLROnPlateauWrapper] = None,
        early_stopping: t.Optional[EarlyStopping] = None,
        early_pruning: t.Optional[EarlyTrialPruning] = None,
        wandb_project: str = "test",
        calc_hessian_for: t.Optional[t.List[str]] = None,
    ) -> None:
        if calc_hessian_for is None:
            calc_hessian_for = ["Training", "Validation", "Testing"]
        self.__attrs_init__(
            window_maker=window_maker,
            file_structure=file_structure,
            dataloader_config=dataloader_config,
            max_epochs_evaluation=max_epochs_evaluation,
            max_epochs_tuning=max_epochs_tuning,
            tuning_trials=tuning_trials,
            early_stopping=early_stopping,
            reduce_on_plateau=reduce_on_plateau,
            early_pruning=early_pruning,
            wandb_project=wandb_project,
            calc_hessian_for=calc_hessian_for,
        )

    def __attrs_post_init__(self) -> None:
        self._save_window_summary()

    def _save_window_summary(self) -> None:
        with open(os.path.join(self.file_structure.path, "window_summary.txt"), "w") as f:
            print(self.window_maker, file=f)

    def run(self, objective: str, collect_and_plot_parameters: bool = False) -> None:
        for inputs, targets, name in self.window_maker():
            stock_dataloader: StockDataloader = StockDataloader(inputs=inputs, target=targets, **self.dataloader_config)
            tuner: Tuner = Tuner(
                dataloader=stock_dataloader,
                file_structure=self.file_structure,
                name=name,
                collect_and_plot_parameters=collect_and_plot_parameters,
                wandb_project=self.wandb_project,
            )
            tuner.tune(
                objective=objective,
                max_epochs_tuning=self.max_epochs_tuning,
                max_epochs_evaluation=self.max_epochs_evaluation,
                trials=self.tuning_trials,
                reduce_on_plateau=self.reduce_on_plateau,
                early_stopping=self.early_stopping,
                early_pruning=self.early_pruning,
            )
            tuner.post_tuning(calc_hessian_for=self.calc_hessian_for)
