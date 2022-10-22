import os
import shutil
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd
import wandb

from fin_dl.plotting.plots import plot_window_metrics
from fin_dl.rolling_pred_study import RollingWindowPrediction
from fin_dl.rolling_predictions.static import StaticWindows
from fin_dl.torch.callbacks import (
    EarlyStopping,
    EarlyTrialPruning,
    ReduceLROnPlateauWrapper,
)
from fin_dl.torch.utils import check_indices, scaled_dot_attention
from fin_dl.utilities import FileStructure, get_dataframes

PATH: str = os.getcwd()
wandb.login(key="")

DAYS_PER_YEAR: int = 365

if __name__ == "__main__":
    companies: t.List[str] = [
        "ARE",
        "AMT",
        "AVB",
        "BXP",
        "CPT",
        "CBRE",
        "CCI",
        "DLR",
        "DRE",
        "EQIX",
        "EQR",
        "ESS",
        "EXR",
        "FRT",
        "PEAK",
        "HST",
        "IRM",
        "KIM",
        "MAA",
        "PLD",
        "PSA",
        "O",
        "REG",
        "SBAC",
        "SPG",
        "UDR",
        "VTR",
        "VNO",
        "WELL",
        "WY",
    ]

    # PSA Symbol
    predict_company: str = "PSA"

    # inputs: pd.DataFrame = pd.read_parquet("/Volumes/Drive (Large)/s_and_p_500_prices_processed.parquet.gzip")
    inputs: pd.DataFrame = pd.read_parquet("s_and_p_500_prices_processed.parquet.gzip")
    # due to computational constraints, the data is trimmed
    start: datetime = pd.to_datetime("01-01-2016", infer_datetime_format=True)
    inputs = inputs.loc[np.isin(inputs.index.get_level_values("symbol"), companies), :].dropna()
    inputs = inputs.loc[inputs.index.get_level_values("date") >= start, :]

    # get target
    target: pd.DataFrame = inputs.loc[:, "diff(log(close))"].to_frame()

    check_indices(inputs, target)

    window_maker: StaticWindows = StaticWindows(
        inputs=inputs,
        targets=target,
        window_width=pd.Timedelta(days=3 * DAYS_PER_YEAR),
        frequency=pd.Timedelta(days=DAYS_PER_YEAR),
    )

    dataloader_config: t.Dict[str, t.Any] = dict(
        batch_size=10,
        date_label="date",
        group_key="symbol",
        scale_target=True,
        sequence_len=5,
        shift_target=1,
        test_set_for_group=predict_company,
        proportions=(0.65, 0.2, 0.15),
        # input_transformation=scaled_dot_attention,
    )
    ID: str = f"master_thesis ({datetime.now().strftime('%Y-%m-%d %H.%M')})"
    file_structure: FileStructure = FileStructure(os.path.join(PATH, ID))
    early_stopping: EarlyStopping = EarlyStopping(monitor="val_naive_r2", direction="max", patience=5, min_delta=0.001)
    reduce_on_plateau: ReduceLROnPlateauWrapper = ReduceLROnPlateauWrapper(
        monitor="val_naive_r2", mode="max", patience=3, verbose=True
    )
    early_pruning: EarlyTrialPruning = EarlyTrialPruning(
        monitor="val_naive_r2",
        direction="max",
        patience=3,
        min_delta=0.001,
        hard_prune_is_close_factor=100.0,
        hard_prune_min_threshold=-2.50,
    )
    rolling_window_predictions: RollingWindowPrediction = RollingWindowPrediction(
        dataloader_config=dataloader_config,
        early_stopping=early_stopping,
        reduce_on_plateau=reduce_on_plateau,
        max_epochs_tuning=5,
        max_epochs_evaluation=100,
        tuning_trials=75,
        file_structure=file_structure,
        early_pruning=early_pruning,
        window_maker=window_maker,
        wandb_project=ID,
        # calculating hessians for each batch is very time intensive and thus limited to the test set
        calc_hessian_for=["Testing"],
    )
    # wandb.init(settings=wandb.Settings(start_method="fork"))
    rolling_window_predictions.run(objective="val_naive_r2", collect_and_plot_parameters=False)

    # post window metrics plot
    windows: pd.DataFrame = get_dataframes(file_structure.predictions_metrics)
    plot_window_metrics(windows, name="window_metrics", saving_path=file_structure.path)

    # zip
    shutil.make_archive(file_structure.path, "zip", file_structure.path)
