import typing as t

import numpy as np
import pandas as pd
from typeguard import typechecked

from fin_dl.utilities import counter, get_values


class CustomIterator:
    data: pd.DataFrame
    date_label: str
    _dates: np.ndarray
    _dates_available: np.ndarray
    _current_date_index: int
    _next_date_index: int
    _date_index_counter: t.Iterator[int]
    _prevent_last_reward_eq_nan: bool

    @typechecked
    def __init__(
        self,
        data: pd.DataFrame,
        date_label: str = "date",
        number_of_data_points: t.Optional[int] = None,
        prevent_last_reward_eq_nan: bool = True,
    ) -> None:
        self.data = data
        self.date_label = date_label
        self._dates = self.__get_dates()
        self._prevent_last_reward_eq_nan = prevent_last_reward_eq_nan
        if number_of_data_points is not None:
            self._dates = (
                self._dates[: (number_of_data_points + 1)]
                if self._prevent_last_reward_eq_nan
                else self._dates[:number_of_data_points]
            )
        # We cannot calculate the reward (financial return) of the last dates (terminal state) stock information as we
        # do not have the next date's stock information available.
        self._dates_available = self._dates[:-1] if self._prevent_last_reward_eq_nan else self._dates

    def __get_dates(self) -> np.ndarray:
        return get_values(frame=self.data, label=self.date_label).sort_values().unique().astype(str).values

    def __iter__(self) -> object:
        self._date_index_counter = counter(length=len(self._dates_available))
        return self

    def __next__(self) -> pd.Series:
        self._current_date_index = next(self._date_index_counter)
        if self._current_date_index < len(self._dates_available):
            current_date: str = self._dates[self._current_date_index]
            query_string: str = f"{self.date_label} == '{current_date}'"
            series: pd.Series = self.data.query(query_string).squeeze().convert_dtypes()
            self._next_date_index = self._current_date_index + 1
            return series
        else:
            raise StopIteration

    def has_next(self) -> bool:
        return self._next_date_index < len(self._dates_available)

    def get_next_values(self) -> pd.Series:
        if self.has_next() or (self._prevent_last_reward_eq_nan and self._next_date_index < len(self._dates)):
            next_date: str = self._dates[self._next_date_index]
            query_string: str = f"{self.date_label} == '{next_date}'"
            return self.data.query(query_string).squeeze().convert_dtypes()
        else:
            return pd.Series(np.repeat(np.nan, repeats=self.data.shape[-1]), index=self.data.columns)

    @property
    def initial_values(self) -> pd.Series:
        start_date: str = self._dates[0]
        query_string: str = f"{self.date_label} == '{start_date}'"
        return self.data.query(query_string).squeeze().convert_dtypes()

    @property
    def current_date(self) -> str:
        return self.dates[self._current_date_index]

    @property
    def dates(self) -> np.ndarray:
        return self._dates_available
