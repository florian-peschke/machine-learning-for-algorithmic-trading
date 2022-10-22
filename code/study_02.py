import copy
import os
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd
import ray
import sklearn
import torch.cuda
from lets_plot import *
from sklearn.preprocessing import QuantileTransformer

from fin_dl import GGPLOT_THEME, SEED
from fin_dl.rl_study import RLStudy
from fin_dl.rl_trading.mdp import MDPConstants
from fin_dl.rl_trading.performance import RLPostTraining, RLPostTrainingMultiAgent
from fin_dl.utilities import init_folder

ray.init(ignore_reinit_error=True)
pd.set_option("display.max_columns", None)
pd.set_option("use_inf_as_na", True)

RAW_DATA_PATH: str = "sp500_prices.parquet"
DATA_WITH_TECHNICAL_INDICATORS: str = "s_and_p_500_prices_processed.parquet.gzip"

PATH: str = init_folder(os.path.join(os.getcwd(), "rl"))

scaler: sklearn.preprocessing = QuantileTransformer(
    n_quantiles=10,
    output_distribution="normal",
    random_state=SEED,
)


def get_raw_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    raw_data: pd.DataFrame = (
        pd.read_parquet(RAW_DATA_PATH).query(f"symbol == '{symbol}'").sort_values(by="date").dropna()
    )
    raw_data["date"] = pd.to_datetime(raw_data["date"], infer_datetime_format=True)
    raw_data = raw_data.sort_values(by=["symbol", "date"]).set_index(["symbol", "date"])

    # inputs with technical indicators
    data_with_technical_indicators: pd.DataFrame = pd.read_parquet(DATA_WITH_TECHNICAL_INDICATORS)

    # merge
    merged: pd.DataFrame = raw_data.merge(
        data_with_technical_indicators,
        left_index=True,
        right_index=True,
    ).dropna()
    return merged.loc[
        (merged.index.get_level_values("date") >= start) & (merged.index.get_level_values("date") < end), :
    ]


def process_data(
    frame: pd.DataFrame, target: str, training_proportion: float
) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.List[str]]:
    frame: pd.DataFrame = copy.deepcopy(frame)
    dates: pd.DatetimeIndex = pd.DatetimeIndex(frame.index.get_level_values(level="date"))
    frame.insert(loc=0, value=frame.loc[:, target], column=f"scaled_{target}")

    # scale all columns apart from the target column
    transform_columns: t.List[str] = [column for column in frame.columns if column != target]

    # training data
    training: pd.DataFrame = frame.loc[dates < np.quantile(a=dates, q=training_proportion), :]
    training = (
        training.loc[:, target]
        .to_frame()
        .merge(
            pd.DataFrame(
                scaler.fit_transform(training[transform_columns]),
                columns=training[transform_columns].columns,
                index=training[transform_columns].index,
            ),
            left_on="date",
            right_on="date",
        )
    )
    # testing data
    testing: pd.DataFrame = frame.loc[dates >= np.quantile(a=dates, q=training_proportion), :]
    testing = (
        frame.loc[:, target]
        .to_frame()
        .merge(
            pd.DataFrame(
                scaler.fit_transform(testing[transform_columns]),
                columns=testing[transform_columns].columns,
                index=testing[transform_columns].index,
            ),
            left_on="date",
            right_on="date",
        )
    )

    return training, testing, transform_columns


if __name__ == "__main__":
    symbol: str = "PSA"
    target_value: str = "close"

    start: datetime = pd.to_datetime("01-01-2016", infer_datetime_format=True)
    end: datetime = pd.to_datetime("01-01-2021", infer_datetime_format=True)

    data: pd.DataFrame = get_raw_data(symbol, start=start, end=end)
    training, testing, transform_columns = process_data(data, target=target_value, training_proportion=0.8)

    study: RLStudy = RLStudy(
        symbol=symbol,
        training_data=training,
        test_data=testing,
        transform_columns=transform_columns,
        additional_config={
            "framework": "torch",
            "horizon": None,
            "num_gpus": torch.cuda.device_count(),
            "hiddens": [256, 128, 64],
            "disable_env_checking": True,
        },
        target_value=target_value,
        training_episodes=250,
    )

    # multi agent evaluation
    multi_agent_evaluation: RLPostTrainingMultiAgent = study.run_multi_agent(return_eval=True)

    # plots
    multi_agent_evaluation.comparison_plot(path=PATH, name="multi_agent_comparison_plot_evaluation")
    multi_agent_evaluation.cum_sum_plot(path=PATH, name="multi_agent_cum_sum_plot_evaluation")
    multi_agent_evaluation.detailed_plot(
        path=PATH, name="multi_agent_detailed_plot_evaluation", sub_plot_weight=1000, subplot_height=300
    )
    multi_agent_evaluation.save_reward_sum(path=PATH)

    # single agent evaluation
    single_agent_evaluation: RLPostTraining = study.run_single_agent(return_eval=True)

    # plots
    single_agent_evaluation.comparison_plot(path=PATH, name="single_agent_comparison_plot_evaluation")
    single_agent_evaluation.cum_sum_plot(path=PATH, name="single_agent_cum_sum_plot_evaluation")
    single_agent_evaluation.detailed_plot(
        path=PATH, name="single_agent_detailed_plot_evaluation", sub_plot_weight=1000, subplot_height=300
    )
    single_agent_evaluation.save_reward_sum(path=PATH)

    # Post comparison plots

    multi: pd.DataFrame = copy.deepcopy(study.cum_sum_multi)
    multi.insert(loc=0, value="multi", column="type")
    multi.to_csv(os.path.join(PATH, "multi_training.csv"))

    single: pd.DataFrame = copy.deepcopy(study.cum_sum_single)
    single.insert(loc=0, value="single", column="type")
    single.to_csv(os.path.join(PATH, "single_training.csv"))

    single_multi: pd.DataFrame = pd.concat([single, multi]).reset_index().set_index(["type", "index"])
    single_multi.to_csv(os.path.join(PATH, "single_multi_training.csv"))

    realized_reward: pd.DataFrame = pd.concat(
        [
            pd.Series(study.cum_sum_multi[MDPConstants.LABEL_REALIZED_REWARD], name="multi"),
            pd.Series(study.cum_sum_single[MDPConstants.LABEL_REALIZED_REWARD], name="single"),
        ],
        axis=1,
    )
    realized_reward.to_csv(os.path.join(PATH, "realized_reward_training.csv"))

    # noinspection PyTypeChecker
    facet_plot = (
        ggplot(
            single_multi.melt(ignore_index=False, var_name="Agent").reset_index().convert_dtypes(),
            aes(x="index", y="value", color="Agent"),
        )
        + geom_line()
        + facet_grid(y="type")
        + labs(title="Single- vs. Multi-Agent", subtitle="Total Reward", x="Episodes", y="", caption="Training")
        + GGPLOT_THEME()
    )
    ggsave(facet_plot, os.path.join(PATH, f"single_vs_multi_all_training.svg"))
    ggsave(facet_plot, os.path.join(PATH, f"single_vs_multi_all_training.html"))

    # noinspection PyTypeChecker
    comparison_plot = (
        ggplot(
            realized_reward.melt(ignore_index=False, var_name="Agent").reset_index().convert_dtypes(),
            aes(x="index", y="value", color="Agent"),
        )
        + geom_line()
        + labs(
            title="Single- vs. Multi-Agent", subtitle="Total Realized Reward", x="Episodes", y="", caption="Training"
        )
        + GGPLOT_THEME()
    )
    ggsave(comparison_plot, os.path.join(PATH, f"single_vs_multi_realized_training.svg"))
    ggsave(comparison_plot, os.path.join(PATH, f"single_vs_multi_realized_training.html"))
