import attrs
import logging
import numpy as np
import pandas as pd
import rpy2.robjects.packages as rpackages
import typing as t
from attr import field
from attrs import define
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import ListVector
from rpy2.robjects.packages import importr, InstalledSTPackage
from tqdm import tqdm

from fin_dl.stock_data.technical_indicators import TechnicalIndicators
from fin_dl.utilities import Bcolors


class OHLCV(t.NamedTuple):
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"

    @property
    def as_list(self) -> t.List[str]:
        return [self.open, self.high, self.low, self.close, self.volume]


class DataFrameInformation(t.NamedTuple):
    date_label: str = "date"
    group_label: str = "symbol"
    sort_by: t.List[str] = [group_label, date_label]
    index: t.List[str] = [group_label, date_label]


def transform_columns(
    frame: pd.DataFrame,
    columns: t.List[str],
    transformation: t.Callable,
    transformation_label: str,
    group_by: t.Optional[str] = None,
    **kwargs,
) -> t.Tuple[t.List[str], pd.DataFrame]:
    data: pd.DataFrame
    if group_by is None:
        data = frame.loc[:, columns].apply(lambda x: transformation(x, **kwargs)).values
    else:
        data = frame.loc[:, columns].groupby(by=group_by).apply(lambda x: transformation(x, **kwargs)).values
    new_columns: t.List[str] = [f"{transformation_label}({column_name})" for column_name in columns]
    transformed_frame: pd.DataFrame = pd.DataFrame(
        data=data,
        columns=new_columns,
        index=frame.index,
    )
    return new_columns, pd.concat([frame, transformed_frame], axis=1)


def add_percentage_change(
    frame: pd.DataFrame, columns: t.List[str], transformation_label: str, group_by: t.Optional[str] = None
) -> pd.DataFrame:
    data: pd.DataFrame
    if group_by is None:
        data = frame.loc[:, columns].pct_change().values
    else:
        data = frame.loc[:, columns].groupby(by=group_by).pct_change().values

    transformed_frame: pd.DataFrame = pd.DataFrame(
        data=data,
        columns=[f"{transformation_label}({column_name})" for column_name in columns],
        index=frame.index,
    )
    return pd.concat([frame, transformed_frame], axis=1)


@define
class AddTechnicalIndicators:
    parquet_path: str

    CRITICAL_VALUES: t.List[t.Union[float, int]] = [0, 0.0, np.nan, pd.NA, np.inf, -np.inf]
    REJECTION_BARRIER_OF_ADF_H0: float = 0.05
    PERCENTAGE_OF_CRITICAL_VALUES_ALLOWED: float = 0.05
    LABEL_DIFFERENCE: str = "diff"
    LABEL_PERCENTAGE_CHANGE: str = "pct_change"
    LABEL_LOGARITHM: str = "log"

    data_frame_information: DataFrameInformation = DataFrameInformation()
    ohlcv: OHLCV = OHLCV()
    r_time_series_package: InstalledSTPackage = field()
    technical_indicators: TechnicalIndicators = TechnicalIndicators()
    max_difference_recursions: int = 7

    @r_time_series_package.default
    def __set_r_wrapper(self) -> None:
        logging.info("Setting R wrapper.")
        utils: InstalledSTPackage = importr("utils")
        if not rpackages.isinstalled("tseries"):
            utils.chooseCRANmirror(ind=1)
            utils.install_packages("tseries")
        return importr("tseries")

    def __get_data(self) -> pd.DataFrame:
        data: pd.DataFrame = pd.read_parquet(self.parquet_path)
        data[self.data_frame_information.date_label] = pd.to_datetime(
            data[self.data_frame_information.date_label], infer_datetime_format=True
        )
        data = data.sort_values(self.data_frame_information.sort_by).set_index(self.data_frame_information.index)
        return data

    def get_processed_data(self) -> pd.DataFrame:
        frame: pd.DataFrame = self.__get_data()
        # Add transformations
        logging.info("Transforming inputs.")
        extended_frame: pd.DataFrame = self.__add_transformations(frame=frame)
        # Add technical indicators
        extended_frame = self.technical_indicators.calculate_all(frame=extended_frame)
        # 1.Time: Drop columns with many critical values
        logging.info("Dropping columns with many critical values.")
        extended_frame = extended_frame.drop(columns=self.__get_columns_with_many_critical_values(frame=extended_frame))
        # Make non-stationary columns stationary
        logging.info("Making inputs stationary.")
        extended_frame = self.__recursive_difference_calculation(frame=extended_frame)
        # 2.Time: Drop columns with many critical values
        extended_frame = extended_frame.drop(columns=self.__get_columns_with_many_critical_values(frame=extended_frame))
        return extended_frame

    def __add_transformations(self, frame: pd.DataFrame) -> pd.DataFrame:
        initial_columns: t.List[str] = frame.columns.tolist()
        # Add percentage change
        frame = add_percentage_change(
            frame=frame, columns=frame.columns.tolist(), transformation_label=self.LABEL_PERCENTAGE_CHANGE
        )
        # Add logarithmic values
        log_columns, frame = transform_columns(
            frame=frame,
            columns=initial_columns,
            transformation=np.log,
            transformation_label=self.LABEL_LOGARITHM,
        )
        # Add first difference of logarithmic and initial columns
        _, frame = transform_columns(
            frame=frame,
            columns=initial_columns + log_columns,
            transformation=np.diff,
            transformation_label=self.LABEL_DIFFERENCE,
            prepend=[np.nan],
        )
        return frame

    def __get_columns_with_many_critical_values(self, frame: pd.DataFrame) -> t.List[str]:
        columns_with_many_critical_values: t.List[str] = []
        for column_name in frame.columns:
            for critical_value in self.CRITICAL_VALUES:
                number_of_critical_values: int = len(frame.loc[frame[column_name] == critical_value, column_name])
                total_number_of_values: int = len(frame.loc[:, column_name])
                percentage: float = number_of_critical_values / total_number_of_values
                if percentage > self.PERCENTAGE_OF_CRITICAL_VALUES_ALLOWED:
                    columns_with_many_critical_values.append(column_name)
                    logging.info(
                        f"Dropped '{column_name}' because it has to many critical values {self.CRITICAL_VALUES}"
                    )
                    break
        return columns_with_many_critical_values

    def __recursive_difference_calculation(
        self,
        frame: pd.DataFrame,
        columns: t.Optional[t.List[str]] = None,
        current_recursion_number: int = 1,
    ) -> pd.DataFrame:
        print(
            f"{Bcolors.BLUE}Difference recursion {current_recursion_number} "
            f"of maximum {self.max_difference_recursions}{Bcolors.RESET}"
        )
        non_stationary_columns: t.List[str] = self.__get_non_stationary_columns(
            frame=(frame if columns is None else frame.loc[:, columns])
        )
        new_column_names: t.List[str] = [
            f"{self.LABEL_DIFFERENCE}({column_name})" for column_name in non_stationary_columns
        ]
        columns_to_transform: t.List[str] = [
            non_stationary_column
            for non_stationary_column, new_column_name in zip(non_stationary_columns, new_column_names)
            if new_column_name not in frame.columns
        ]
        adjusted_and_extended_frame: pd.DataFrame
        added_diff_columns, adjusted_and_extended_frame = transform_columns(
            frame=frame,
            columns=columns_to_transform,
            transformation=np.diff,
            transformation_label=self.LABEL_DIFFERENCE,
            prepend=[np.nan],
        )
        logging.info(f"Dropping Non-stationary columns: {non_stationary_columns}")
        new_frame: pd.DataFrame = adjusted_and_extended_frame.drop(columns=non_stationary_columns)
        if self.max_difference_recursions == current_recursion_number or len(added_diff_columns) == 0:
            if len(added_diff_columns) == 0:
                print(
                    f"{Bcolors.BLUE}Recursion stopped after {current_recursion_number} recursions because no "
                    f"non-stationary columns left.{Bcolors.RESET}"
                )
            return new_frame
        return self.__recursive_difference_calculation(
            frame=new_frame, columns=added_diff_columns, current_recursion_number=current_recursion_number + 1
        )

    def __get_non_stationary_columns(self, frame: pd.DataFrame) -> t.List[str]:
        non_stationary_columns: t.List[str] = []
        for column_name in tqdm(frame.columns, desc="Augmented-Dickey-Fuller Tests"):
            for group in frame.loc[:, column_name].groupby(by=self.data_frame_information.group_label):
                if not self.__is_stationary(series=group[1]):
                    logging.info(f"'{column_name}' is not stationary.")
                    non_stationary_columns.append(column_name)
                    break
        return non_stationary_columns

    def __is_stationary(self, series: pd.Series) -> bool:
        clean_series: pd.Series
        with pd.option_context("use_inf_as_na", True):
            clean_series = series[np.isin(element=series, test_elements=self.CRITICAL_VALUES, invert=True)].dropna()
        try:
            adt_results: ListVector = self.r_time_series_package.adf_test(clean_series.tolist())
        except RRuntimeError:
            print(clean_series)
            return False
        output: t.Dict[str, t.Any] = {name: value[0] for name, value in zip(adt_results.names, list(adt_results))}
        p_value = output["p.value"]
        return p_value <= self.REJECTION_BARRIER_OF_ADF_H0
