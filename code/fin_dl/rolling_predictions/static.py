import typing as t
from datetime import datetime

import pandas as pd
from attr import define, field

from fin_dl.torch.utils import check_indices
from fin_dl.utilities import get_values


@define
class StaticWindows:
    inputs: pd.DataFrame
    targets: pd.DataFrame

    date_label: str
    window_width: pd.Timedelta
    frequency: pd.Timedelta
    date_format: str

    _dates: pd.DatetimeIndex = field(init=False)

    def __init__(
        self,
        inputs: pd.DataFrame,
        targets: pd.DataFrame,
        date_label: str = "date",
        window_width: pd.Timedelta = pd.Timedelta(days=180),
        frequency: pd.Timedelta = pd.Timedelta(days=180),
        date_format: str = "%YM%m",
    ) -> None:
        check_indices(inputs, targets)
        self.__attrs_init__(
            inputs=inputs,
            targets=targets,
            date_label=date_label,
            window_width=window_width,
            frequency=frequency,
            date_format=date_format,
        )

    def __attrs_post_init__(self) -> None:
        self._dates = pd.DatetimeIndex(get_values(frame=self.inputs, label=self.date_label).sort_values().unique())

    def _yield_start_dates(self) -> t.Iterator[datetime]:
        current: pd.Timestamp = self._dates.min()
        while current < self._dates.max() - self.window_width:
            yield current
            current += self.frequency

    def _get_window(self, frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        dates: pd.DatetimeIndex = pd.DatetimeIndex(get_values(frame=frame, label=self.date_label))
        return frame.loc[(start <= dates) & (dates < end), :]

    def _get_window_frames(self, start: datetime, end: datetime) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        return self._get_window(self.inputs, start=start, end=end), self._get_window(self.targets, start=start, end=end)

    def _format_date(self, date: datetime) -> str:
        return date.strftime(self.date_format)

    def _date_description(self, start: datetime, end: datetime) -> str:
        return f"[{self._format_date(start)}, {self._format_date(end)})"

    def __call__(self) -> t.Iterator[t.Tuple[pd.DataFrame, pd.DataFrame, str]]:
        for start, end in zip(self.start_dates, self.end_dates):
            inputs, targets = self._get_window_frames(start=start, end=end)
            check_indices(inputs, targets)
            yield inputs, targets, self._date_description(start, end)

    def __repr__(self) -> str:
        return "\n".join(
            [
                "\n".join(
                    f"{(i + 1):>4} | {self._date_description(start, end)}"
                    for i, (start, end) in enumerate(zip(self.start_dates, self.end_dates))
                ),
                f"\nWindow width: {self.window_width}",
                f"Frequency: {self.frequency}",
            ]
        )

    @property
    def start_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(list(self._yield_start_dates()))

    @property
    def end_dates(self) -> pd.DatetimeIndex:
        return self.start_dates + self.window_width
