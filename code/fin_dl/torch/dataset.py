import typing as t
from abc import ABC

import numpy as np
import pandas as pd
import sklearn
import torch
from attr import define, field
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from torch.utils.data import DataLoader, Dataset
from typeguard import typechecked

from fin_dl import SEED
from fin_dl.torch import device, TENSOR_DTYPE
from fin_dl.torch.utils import check_indices, check_len_eq, check_shift_dates, TimeSeriesSplit, unity
from fin_dl.utilities import get_values


@define
class TensorTransformation:
    pipeline: t.List[t.Any] = []

    def append(self, *args) -> None:
        for arg in args:
            self.pipeline.append(arg)

    def transform(self, inputs: t.Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        for transformer in self.pipeline:
            inputs = transformer(inputs)
        return inputs

    def __call__(self, inputs: t.Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.transform(inputs)

    @property
    def is_empty(self) -> bool:
        return len(self.pipeline) == 0


@define
class ToTorchTensor:
    def __call__(self, inputs: np.ndarray) -> torch.Tensor:
        return getattr(torch.from_numpy(inputs), TENSOR_DTYPE)()


class ToDevice:
    def __init__(self):
        pass

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.to(device)


@define
class StockDataset(Dataset, ABC):
    inputs: np.ndarray
    target: np.ndarray
    sequence_len: int
    target_as_sequence: bool
    input_transformation: t.Callable[[np.ndarray], np.ndarray]
    transformations: TensorTransformation = TensorTransformation()

    @typechecked
    def __init__(
        self,
        inputs: t.Union[np.ndarray, pd.DataFrame],
        target: t.Union[np.ndarray, pd.DataFrame],
        sequence_len: int,
        target_as_sequence: bool = False,
        input_transformation: t.Callable[[np.ndarray], np.ndarray] = unity,
        transformations: TensorTransformation = TensorTransformation(),
    ):
        inputs = inputs.values if isinstance(inputs, pd.DataFrame) else inputs
        target = target.values if isinstance(target, pd.DataFrame) else target
        if transformations.is_empty:
            transformations.append(ToTorchTensor(), ToDevice())
        self.__attrs_init__(
            inputs=inputs,
            target=target,
            sequence_len=sequence_len,
            target_as_sequence=target_as_sequence,
            input_transformation=input_transformation,
            transformations=transformations,
        )

    def __attrs_post_init__(self):
        if len(self.inputs) != len(self.target):
            raise ValueError(
                f"The length of the inputs {len(self.inputs)} and the target_label {len(self.target)} data differs."
            )

    def _get_slice(self, index: int) -> slice:
        return slice(index, index + self.sequence_len)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        if self.target_as_sequence:
            return (
                self.transformations(self.input_transformation(self.inputs[self._get_slice(index)])),
                self.transformations(self.target[self._get_slice(index)]),
            )

        """
        reason to subtract 1: 
        
        import numpy as np
    
        index = 0
        seq_len = 4
        
        np.arange(10)[slice(index, seq_len)]
        >>> array([0, 1, 2, 3])
        np.arange(10)[index + seq_len]
        >>> 4
        
        but it must be 3 => adjust indexing:
        np.arange(10)[index + seq_len - 1]
        >>> 3
        """
        return (
            self.transformations(self.input_transformation(self.inputs[self._get_slice(index)])),
            self.transformations(
                self.target[index + self.sequence_len - 1],
            ),
        )

    def __len__(self) -> int:
        return len(self.target) - self.sequence_len + 1


@define(slots=False)
class StockDataloader:
    inputs: pd.DataFrame
    target: pd.DataFrame
    batch_size: int
    date_label: str
    group_key: str
    input_transformation: t.Callable[[np.ndarray], np.ndarray]
    proportions: t.Tuple[float, float, float]
    scale_inputs: bool
    scale_target: bool
    scaler: sklearn.preprocessing
    shift_target: int
    shuffle: bool
    sequence_len: int
    target_as_sequence: bool
    drop_last: bool
    test_set_for_group: t.Optional[str]

    _dataset: TimeSeriesSplit = field(init=False)

    @typechecked
    def __init__(
        self,
        inputs: pd.DataFrame,
        target: pd.DataFrame,
        batch_size: int,
        date_label: str,
        group_key: str,
        scale_target: bool,
        shift_target: int,
        sequence_len: int,
        scaler: sklearn.preprocessing = QuantileTransformer(
            n_quantiles=10,
            output_distribution="normal",
            random_state=SEED,
        ),
        input_transformation: t.Callable[[np.ndarray], np.ndarray] = unity,
        proportions: t.Tuple[float, float, float] = (0.7, 0.15, 0.15),
        scale_inputs: bool = True,
        shuffle: bool = False,
        drop_last: bool = True,
        target_as_sequence: bool = False,
        test_set_for_group: t.Optional[str] = None,
    ):
        self.__attrs_init__(
            inputs=inputs,
            target=target,
            sequence_len=sequence_len,
            scale_inputs=scale_inputs,
            input_transformation=input_transformation,
            scale_target=scale_target,
            batch_size=batch_size,
            shuffle=shuffle,
            shift_target=shift_target,
            date_label=date_label,
            group_key=group_key,
            scaler=scaler,
            drop_last=drop_last,
            proportions=proportions,
            target_as_sequence=target_as_sequence,
            test_set_for_group=test_set_for_group,
        )
        super().__init__()

    def __attrs_post_init__(self) -> None:
        check_indices(self.inputs, self.target)

    def _scale(self, frame: pd.DataFrame) -> t.Iterator[pd.DataFrame]:
        for group in frame.groupby(by=self.group_key):
            dates: pd.DatetimeIndex = pd.DatetimeIndex(get_values(frame=group[-1], label=self.date_label))
            training_data: pd.DataFrame = group[-1].loc[dates < np.quantile(a=dates, q=self.proportions[0]), :]
            self.scaler.fit(training_data)
            yield pd.DataFrame(self.scaler.transform(group[-1]), columns=group[-1].columns, index=group[-1].index)

    def prepare_data(self) -> None:

        inputs: pd.DataFrame = self.inputs
        if self.scale_inputs:
            # scale inputs
            inputs: pd.DataFrame = pd.concat(list(self._scale(inputs)))

        target: pd.DataFrame = self.target
        if self.scale_target:
            # scale target
            target = pd.concat(list(self._scale(target)))

        # shift inputs
        inputs: pd.DataFrame = pd.concat(list(self._shift(frame=inputs, indexing=slice(None, -self.shift_target))))
        target: pd.DataFrame = pd.concat(list(self._shift(frame=target, indexing=slice(self.shift_target, None))))

        # validation checks
        check_shift_dates(inputs, target)
        check_len_eq(inputs, target)

        # split inputs
        self._dataset = TimeSeriesSplit(
            inputs=inputs,
            target=target,
            splitting_proportions=self.proportions,
            group_key=self.group_key,
            date_label=self.date_label,
        )

    def _shift(self, frame: pd.DataFrame, indexing: slice) -> t.Iterator[pd.DataFrame]:
        for group in frame.groupby(by=self.group_key):
            yield group[-1].iloc[indexing, :]

    def train_dataloader(self) -> DataLoader:
        return self.training

    def val_dataloader(self) -> DataLoader:
        return self.validation

    def test_dataloader(self) -> DataLoader:
        return self.testing

    def predict_dataloader(self) -> DataLoader:
        return self.testing

    def _get_data_loader(self, dataset: t.Tuple[pd.DataFrame, pd.DataFrame]) -> DataLoader:
        dataset: StockDataset = StockDataset(
            *dataset,
            sequence_len=self.sequence_len,
            target_as_sequence=self.target_as_sequence,
            input_transformation=self.input_transformation,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last
            # num_workers=NUMBER_OF_WORKERS,
        )

    @property
    def dataset(self) -> TimeSeriesSplit:
        try:
            return self._dataset
        except AttributeError:
            self.prepare_data()
            return self._dataset

    @property
    def training(self) -> DataLoader:
        return self._get_data_loader(self.dataset.training)

    @property
    def validation(self) -> DataLoader:
        return self._get_data_loader(self.dataset.validation)

    @property
    def testing(self) -> DataLoader:
        if self.test_set_for_group is None:
            return self._get_data_loader(self.dataset.testing)
        return self._get_data_loader(
            self.dataset.get_group_set(*self.dataset.testing, group_key=self.group_key, group=self.test_set_for_group)
        )

    @property
    def predicting(self) -> DataLoader:
        if self.test_set_for_group is None:
            return self._get_data_loader(self.dataset.testing)
        return self._get_data_loader(
            self.dataset.get_group_set(group_key=self.group_key, group=self.test_set_for_group, *self.dataset.testing)
        )

    @property
    def input_size(self) -> int:
        return self.inputs.shape[-1]

    @property
    def output_size(self) -> int:
        return self.target.shape[-1]

    @property
    def features(self) -> t.List[str]:
        return self.inputs.columns.tolist()
