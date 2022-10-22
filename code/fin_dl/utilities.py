from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import optuna
import os
import pandas as pd
import pickle
import re
import tabulate
import time
import typing as t
import yaml
from attr import define
from datetime import timedelta
from shutil import rmtree
from typeguard import typechecked


class MainIndicatorAndTransformation(t.NamedTuple):
    """
    NamedTuple to store the `main indicator` (string) and the `transformation` (string).
    """

    main_indicator: str
    transformation: str


def save_pickle(filename: str, object: t.Any, path: str):
    """
    Serialize object.

    Parameters
    ----------
    filename
    object
    path

    Returns
    -------

    """
    with open(f"{path}/{filename}.pkl", "wb") as file:
        pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str, path: str):
    """
    Load serialized object.
    """
    with open(f"{path}/{filename}.pkl", "rb") as file:
        # noinspection PickleLoad
        saved_model = pickle.load(file)
    return saved_model


def get_numeric_columns(data: pd.DataFrame) -> t.Generator[bool, None, None]:
    """
    Return a generator that contains only booleans => whether the respective element/column in `inputs`
    is of type `numeric`.
    """
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]) and not pd.api.types.is_bool_dtype(data[col]):
            yield True
        else:
            yield False


def get_dataframe_time_freq(data: pd.DataFrame, date_index_label: str) -> str:
    dates: pd.Index = data.index.get_level_values(date_index_label).unique().sort_values()
    dates_padded = pd.date_range(start=dates[0], end=dates[-1])
    return pd.infer_freq(dates_padded)


def split_column_name(column_names: str) -> MainIndicatorAndTransformation:
    """
    Split the column names and return for each one the named Tuple `MainIndicatorAndTransformation`.
    """
    regex: None | re.Match = re.search(r"^(?P<transformation>.*)\((?P<indicator>.*)\)$", column_names)
    if regex is None:
        return MainIndicatorAndTransformation(main_indicator=column_names, transformation="")
    else:
        return MainIndicatorAndTransformation(
            main_indicator=regex.groupdict()["indicator"], transformation=regex.groupdict()["transformation"]
        )


def sort_columns_by_main_indicator(data: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the dataframe according to the main `indicator` (primary sort)
    and the applied transformations `transformation` (secondary sort).
    """
    return data.loc[:, sorted(data.columns, key=lambda column: split_column_name(column))]


class Bcolors:
    BLUE = "\033[94m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def stop_time(func):
    def dec(*args, **kwargs):
        start: float = time.time()
        output: t.Any = func(*args, **kwargs)
        stop: float = time.time()
        print(f"Time: {Bcolors.BLUE}{timedelta(seconds=stop - start)}{Bcolors.RESET}\n")
        return output

    return dec


def get_values(frame: pd.DataFrame, label: str) -> t.Union[pd.Index, pd.DatetimeIndex, pd.Series]:
    try:
        return frame.loc[:, label]
    except KeyError:
        return frame.index.get_level_values(level=label)


class ValueSpace:
    min: t.Union[float, int]
    max: t.Union[float, int]
    cum_sum: t.Optional[t.Union[float, int]]

    @typechecked
    def __init__(
        self, min: t.Union[float, int], max: t.Union[float, int], cum_sum: t.Optional[t.Union[float, int]] = None
    ) -> None:
        self.min = min
        self.max = max
        self.cum_sum = cum_sum

    def check_value_space(self, value: t.Union[float, int]) -> t.Union[float, int]:

        if self.min <= value <= self.max:
            return value
        else:
            raise ValueError(f"Value must be bounded by the closed interval [{self.min}, {self.max}]!")

    def check_cum_sum(self, values: t.List[t.Union[float, int]] or np.ndarray) -> None:
        cum_sum: np.ndarray = np.sum(values)
        if cum_sum != self.cum_sum:
            raise ValueError(f"Values sum up to {cum_sum}, but must sum up to exactly {self.cum_sum}!")


@typechecked
def get_dataframe_time_freq(data: pd.DataFrame, date_index_label: str) -> str:
    """

    Parameters
    ----------
    data
    date_index_label

    Returns
    -------

    """
    dates: np.ndarray = np.sort(np.unique(get_values(frame=data, label=date_index_label)))
    dates_padded = pd.date_range(start=dates[0], end=dates[-1])
    return pd.infer_freq(dates_padded)


@typechecked
def pad_array(array: t.Union[t.List, np.ndarray], length: int, value: t.Any = np.nan) -> t.Union[t.List, np.ndarray]:
    if len(array) >= length:
        return array
    return np.concatenate((array, np.repeat(value, length - len(array))), axis=None)


@typechecked
def regex_group_search(text: str, regex: str, group_key: str) -> str:
    try:
        return re.search(regex, text).groupdict()[group_key]
    except AttributeError:
        return text


@typechecked
def create_dir_if_missing(dir: str, return_dir: bool = True) -> t.Optional[str]:
    if not os.path.exists(dir):
        os.makedirs(dir)
    if return_dir:
        return dir


@typechecked
def counter(length: int) -> t.Generator[int, None, None]:
    yield from range(length)


def calc_golden_ratio(a: t.Optional[float or int] = None, b: t.Optional[float or int] = None) -> float:
    """
    Calculate the golden ratio for the given value.
    a/b = gr, with gr ~= 1.618
    """
    gr: float = (1 + 5 ** (1 / 2)) / 2
    if a is None:
        return b * gr
    elif b is None:
        return a / gr
    else:
        raise ValueError("a and b are both None!")


def series_to_frame_with_names_as_separate_column(
    series: pd.Series, value_col_name: str = "value", name_col_name: str = "name", index_name: str = "index"
) -> pd.DataFrame:
    name_data_frame: pd.DataFrame = pd.DataFrame(
        np.repeat(str(series.name), repeats=len(series)), index=series.index, columns=[name_col_name]
    )
    series_as_data_frame: pd.DataFrame = series.to_frame(name=value_col_name)
    merged_data_frames: pd.DataFrame = pd.merge(
        left=series_as_data_frame, right=name_data_frame, left_index=True, right_index=True
    )
    merged_data_frames.index.name = index_name
    return merged_data_frames


def init_folder(path: str) -> str:
    rmtree(path, ignore_errors=True)
    return create_dir_if_missing(dir=path)


def concat_parquet(path: str, find: str = "*.parquet") -> pd.DataFrame:
    return dd.read_parquet(os.path.join(path, find), engine="pyarrow").compute()


def save_optuna_plot(
    path: str,
    study: optuna.Study,
    name: str,
    plot_fun: t.Callable = optuna.visualization.plot_param_importances,
    plot_kwargs: t.Optional[dict] = None,
) -> None:
    if plot_kwargs is None:
        plot_kwargs = {}
    try:
        plot_fun(study, **plot_kwargs).write_image(os.path.join(path, f"{name}.pdf"))
    except BaseException as e:
        with open(os.path.join(path, f"{name}.txt"), "w") as f:
            print(e, file=f)


def frame_to_latex(frame: pd.DataFrame, path: str, name: str, **kwargs) -> None:
    with open(os.path.join(path, f"{name}.txt"), "w") as f:
        print(tabulate.tabulate(frame, headers="keys", tablefmt="latex", showindex=True, **kwargs), file=f)


@typechecked
def to_pickle(object_: t.Any, filename: str):
    with open(filename, "wb") as file:
        pickle.dump(object_, file, pickle.HIGHEST_PROTOCOL)


@typechecked
def from_pickle(filename: str):
    with open(filename, "rb") as file:
        # noinspection PickleLoad
        return pickle.load(file)


def to_yaml(dictionary: t.Dict[str, t.Any], filename: str) -> None:
    with open(filename, "w") as file:
        file.write(yaml.dump(dictionary))


@typechecked
def from_yaml(filename: str) -> t.Dict[str, t.Any]:
    with open(filename, "r") as file:
        return yaml.safe_load(file)


@define
class FileStructure:
    path: str

    def __init__(self, path: str) -> None:
        self.__attrs_init__(path=init_folder(path))

    @property
    def best_model(self) -> str:
        return create_dir_if_missing(os.path.join(self.path, "best_model"))

    @property
    def best_model_state_dict(self) -> str:
        return create_dir_if_missing(os.path.join(self.best_model, "state_dict"))

    @property
    def best_model_parameters(self) -> str:
        return create_dir_if_missing(os.path.join(self.best_model, "parameters"))

    @property
    def best_model_parameters_weight_grads(self) -> str:
        return create_dir_if_missing(os.path.join(self.best_model_parameters, "weight_grads"))

    @property
    def best_model_parameters_weight(self) -> str:
        return create_dir_if_missing(os.path.join(self.best_model_parameters, "weight"))

    @property
    def best_model_parameters_bias(self) -> str:
        return create_dir_if_missing(os.path.join(self.best_model_parameters, "bias"))

    @property
    def best_model_parameters_bias_grads(self) -> str:
        return create_dir_if_missing(os.path.join(self.best_model_parameters, "bias_grads"))

    @property
    def best_model_architecture(self) -> str:
        return create_dir_if_missing(os.path.join(self.best_model, "architecture"))

    @property
    def predictions(self) -> str:
        return create_dir_if_missing(os.path.join(self.path, "predictions"))

    @property
    def predictions_metrics(self) -> str:
        return create_dir_if_missing(os.path.join(self.predictions, "metrics"))

    @property
    def predictions_metrics_batch_history(self) -> str:
        return create_dir_if_missing(os.path.join(self.predictions, "metrics_batch_history"))

    @property
    def predictions_values(self) -> str:
        return create_dir_if_missing(os.path.join(self.predictions, "values"))

    @property
    def fitting_history(self) -> str:
        return create_dir_if_missing(os.path.join(self.path, "fitting_history"))

    @property
    def feature_importance(self) -> str:
        return create_dir_if_missing(os.path.join(self.path, "feature_importance"))

    @property
    def feature_importance_jacobian(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance, "jacobian"))

    @property
    def feature_importance_jacobian_variance(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_jacobian, "variance"))

    @property
    def feature_importance_jacobian_signed_mean(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_jacobian, "signed_mean"))

    @property
    def feature_importance_jacobian_signed_mean_corr(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_jacobian, "signed_mean_corr"))

    @property
    def feature_importance_jacobian_positive_valued_mean(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_jacobian, "positive_valued_mean"))

    @property
    def feature_importance_jacobian_rank(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_jacobian, "rank"))

    @property
    def feature_importance_hessian(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance, "hessian"))

    @property
    def feature_importance_hessian_variance(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_hessian, "variance"))

    @property
    def feature_importance_hessian_signed_mean(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_hessian, "signed_mean"))

    @property
    def feature_importance_hessian_positive_valued_mean(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_hessian, "positive_valued_mean"))

    @property
    def feature_importance_hessian_diagonals_variance(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_hessian, "diagonals_variance"))

    @property
    def feature_importance_hessian_diagonals_signed_mean(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_hessian, "diagonals_signed_mean"))

    @property
    def feature_importance_hessian_diagonals_signed_mean_corr(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_hessian, "signed_mean_corr"))

    @property
    def feature_importance_hessian_diagonals_positive_valued_mean(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_hessian, "diagonals_positive_valued_mean"))

    @property
    def feature_importance_hessian_rank(self) -> str:
        return create_dir_if_missing(os.path.join(self.feature_importance_hessian, "rank"))

    @property
    def tuning_data(self) -> str:
        return create_dir_if_missing(os.path.join(self.path, "tuning_data"))

    @property
    def tuning_data_plots(self) -> str:
        return create_dir_if_missing(os.path.join(self.tuning_data, "plots"))

    @property
    def tuning_data_plots_hyperparameter_importance(self) -> str:
        return create_dir_if_missing(os.path.join(self.tuning_data_plots, "hyperparameter_importance"))

    @property
    def tuning_data_plots_plot_slice(self) -> str:
        return create_dir_if_missing(os.path.join(self.tuning_data_plots, "plot_slice"))

    @property
    def tuning_data_study(self) -> str:
        return create_dir_if_missing(os.path.join(self.tuning_data, "study"))

    @property
    def tuning_data_best_trial(self) -> str:
        return create_dir_if_missing(os.path.join(self.tuning_data, "best_trial"))

    @property
    def tuning_data_trials(self) -> str:
        return create_dir_if_missing(os.path.join(self.tuning_data, "trials"))


def float_to_str(val: float, to_scientific: int = 5) -> str:
    if 10 ** -to_scientific < val < 10 ** to_scientific or -(10 ** to_scientific) < val < -(10 ** -to_scientific):
        return f"{val:.4f}" if -1.0 < val < 1.0 else f"{val:.2f}"
    return f"{val:.2E}"


def get_dataframes(data_path: str) -> pd.DataFrame:
    container: t.List[pd.DataFrame] = []

    for path, _, files in os.walk(data_path):
        for file in files:
            parts: t.Dict[str, str] = re.search(r"^(?P<name>.*)\.(?P<file_type>.*)$", file).groupdict()
            name: str = parts["name"]
            file_type: str = parts["file_type"]
            if file_type == "csv":
                frame: pd.DataFrame = pd.read_csv(os.path.join(path, file))
                frame.index = [name]
                try:
                    container.append(frame.drop(columns="Unnamed: 0"))
                except KeyError:
                    container.append(frame)
    return pd.concat(container)
