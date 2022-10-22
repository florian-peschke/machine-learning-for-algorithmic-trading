import os
import typing as t

import pandas as pd
import torch
from attr import define, field
from torch import nn
from torch.autograd.functional import hessian, jacobian
from torch.utils.data import DataLoader
from tqdm import tqdm
from typeguard import typechecked

from fin_dl.plotting.plots import bar_plot, heat_map
from fin_dl.torch import device
from fin_dl.torch.dataset import StockDataloader
from fin_dl.torch.utils import IncrementalSecondMoment
from fin_dl.utilities import FileStructure


@define
class ModelInsights:
    dataloader: DataLoader
    input_size: int
    model: nn.Module

    def __init__(self, model: nn.Module, dataloader: DataLoader, input_size: int) -> None:
        self.__attrs_init__(model=model, dataloader=dataloader, input_size=input_size)

    def jacobian(self) -> IncrementalSecondMoment:
        # init the second moment function for the jacobian
        moment: IncrementalSecondMoment = IncrementalSecondMoment((1, self.input_size)).to(device)

        # turn evaluation mode on
        self.model.eval()

        # calculate jacobian for each batch
        for x, _ in tqdm(self.dataloader, desc="Calculating Second Moment of Jacobians"):
            # noinspection PyTypeChecker
            jac: torch.Tensor = jacobian(self.model.forward, x).detach()

            jac = torch.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
            # noinspection PyTypeChecker
            if torch.all(jac == 0.0):
                continue

            # apply einsum
            moment.update(torch.einsum("...bij->bij", jac))

        return moment

    # @torchsnooper.snoop()
    def hessian(self) -> IncrementalSecondMoment:
        # init the second moment function for the hessian
        moment: IncrementalSecondMoment = IncrementalSecondMoment((self.input_size, self.input_size)).to(device)

        # turn evaluation mode on
        self.model.eval()

        # calculate jacobian for each batch
        for x, _ in tqdm(self.dataloader, desc="Calculating Second Moment of Hessians"):
            container: t.List[torch.Tensor] = []

            # calculate the hessian for each batch element, which is of size (seq_len, input_size)
            for batch_element in x:
                # noinspection PyTypeChecker
                hess: torch.Tensor = hessian(self.model.forward, batch_element.view(1, x.size(-2), x.size(-1))).detach()

                hess = torch.nan_to_num(hess, nan=0.0, posinf=0.0, neginf=0.0)
                # noinspection PyTypeChecker
                if torch.all(hess == 0.0):
                    continue

                # apply einsum
                container.append(torch.einsum("abcdef->bcf", hess))

            # stack containers and update moment
            if len(container) != 0:
                batch_hess: torch.Tensor = torch.stack(container, dim=0)
                moment.update(batch=batch_hess.view(-1, batch_hess.size(-2), batch_hess.size(-1)))

        return moment


@define
class PlotModelInsights:
    model: nn.Module
    datasets: StockDataloader
    file_structure: FileStructure
    name: str
    calc_hessian_for: t.List[str]

    _jacobian_partitions_signed_mean: t.List[pd.DataFrame] = field(init=False)
    _hessian_diagonals_partitions_signed_mean: t.List[pd.DataFrame] = field(init=False)

    @typechecked
    def __init__(
        self,
        model: nn.Module,
        datasets: StockDataloader,
        file_structure: FileStructure,
        name: str,
        calc_hessian_for: t.Optional[t.List[str]] = None,
    ) -> None:
        if calc_hessian_for is None:
            calc_hessian_for = ["Training", "Validation", "Testing"]
        self.__attrs_init__(
            model=model, datasets=datasets, file_structure=file_structure, name=name, calc_hessian_for=calc_hessian_for
        )

    def __attrs_post_init__(self) -> None:
        self._reset_container()
        self._create()

    def _reset_container(self) -> None:
        self._jacobian_partitions_signed_mean = []
        self._hessian_diagonals_partitions_signed_mean = []

    def _create(self) -> None:
        for description, dataloader in zip(
            ["Training", "Validation", "Testing"],
            [self.datasets.training, self.datasets.validation, self.datasets.testing],
        ):
            model_insights: ModelInsights = ModelInsights(
                model=self.model, dataloader=dataloader, input_size=self.datasets.input_size
            )

            # compute jacobians
            jac: IncrementalSecondMoment = model_insights.jacobian()

            # plot
            self._plot_jacobian(jac, description=description)

            # append to list
            self._jacobian_partitions_signed_mean.append(
                pd.DataFrame(jac.signed_mean.view(-1, 1), index=self.datasets.features, columns=[description])
            )

            # check if hessian should be calculated
            if description in self.calc_hessian_for:
                # calc hessian
                hess: IncrementalSecondMoment = model_insights.hessian()

                # plot
                self._plot_hessian(hess, description=description)

                # append to list
                self._hessian_diagonals_partitions_signed_mean.append(
                    pd.DataFrame(
                        torch.diagonal(hess.signed_mean).view(-1, 1),
                        index=self.datasets.features,
                        columns=[description],
                    )
                )

        # plot corr
        self._plot_partition_corr()
        self._reset_container()

    # noinspection PyUnboundLocalVariable
    def _plot_partition_corr(self) -> None:
        # jacobian
        jacobian_partitions_signed_mean_corr: pd.DataFrame = pd.concat(
            self._jacobian_partitions_signed_mean, axis=1
        ).corr()
        jacobian_partitions_signed_mean_corr.to_csv(
            os.path.join(
                self.file_structure.feature_importance_jacobian_signed_mean_corr,
                f"{self.name}.csv",
            )
        )
        try:
            heat_map(
                jacobian_partitions_signed_mean_corr,
                path=self.file_structure.feature_importance_jacobian_signed_mean_corr,
                name=self.name,
                title_args=dict(label="Correlation", subtitle="Jacobians"),
                tiles_and_points_args=dict(type="lower", diag=False),
            )
        except TypeError as e:
            with open(
                os.path.join(
                    self.file_structure.feature_importance_jacobian_signed_mean_corr,
                    f"{self.name}.txt",
                ),
                "w",
            ) as f:
                print("\n".join([str(e), jacobian_partitions_signed_mean_corr]), file=f)

        # check if hessian should be calculated
        if len(self.calc_hessian_for) > 1:
            # hessian
            hessian_diagonals_partitions_signed_mean_corr: pd.DataFrame = pd.concat(
                self._hessian_diagonals_partitions_signed_mean, axis=1
            ).corr()
            hessian_diagonals_partitions_signed_mean_corr.to_csv(
                os.path.join(
                    self.file_structure.feature_importance_hessian_diagonals_signed_mean_corr,
                    f"{self.name}.csv",
                )
            )
            try:
                heat_map(
                    hessian_diagonals_partitions_signed_mean_corr,
                    path=self.file_structure.feature_importance_hessian_diagonals_signed_mean_corr,
                    name=self.name,
                    title_args=dict(label="Correlation", subtitle="diag(Hessians)"),
                    tiles_and_points_args=dict(type="lower", diag=False),
                )
            except TypeError as e:
                with open(
                    os.path.join(
                        self.file_structure.feature_importance_hessian_diagonals_signed_mean_corr,
                        f"{self.name}.txt",
                    ),
                    "w",
                ) as f:
                    print("\n".join([str(e), hessian_diagonals_partitions_signed_mean_corr]), file=f)

    def _plot_jacobian(self, jac: IncrementalSecondMoment, description: str) -> None:
        shared_args: t.Dict[str, dict] = dict(
            aes_args=dict(x="value", y="variable"), name=f"{self.name}_{description.lower()}"
        )

        # variance
        variance: pd.DataFrame = (
            pd.DataFrame(jac.compute(), columns=self.datasets.features).melt().reset_index().sort_values("value")
        )
        variance.to_csv(
            os.path.join(
                self.file_structure.feature_importance_jacobian_variance, f"{self.name}_{description.lower()}.csv"
            )
        )
        bar_plot(
            variance,
            **shared_args,
            path=self.file_structure.feature_importance_jacobian_variance,
            labs_args=dict(title="Variance of Jacobians", subtitle=self.name, caption=description, x="", y=""),
        )

        # signed_mean
        signed_mean: pd.DataFrame = (
            pd.DataFrame(jac.signed_mean, columns=self.datasets.features).melt().reset_index().sort_values("value")
        )
        signed_mean.to_csv(
            os.path.join(
                self.file_structure.feature_importance_jacobian_signed_mean, f"{self.name}_{description.lower()}.csv"
            )
        )
        bar_plot(
            signed_mean,
            **shared_args,
            path=self.file_structure.feature_importance_jacobian_signed_mean,
            labs_args=dict(title="Mean of Jacobians", subtitle=self.name, caption=description, x="", y=""),
        )

        # only positive mean
        positive_valued_mean: pd.DataFrame = (
            pd.DataFrame(jac.positive_valued_mean, columns=self.datasets.features)
            .melt()
            .reset_index()
            .sort_values("value")
        )
        positive_valued_mean.to_csv(
            os.path.join(
                self.file_structure.feature_importance_jacobian_positive_valued_mean,
                f"{self.name}_{description.lower()}.csv",
            )
        )
        bar_plot(
            positive_valued_mean,
            **shared_args,
            path=self.file_structure.feature_importance_jacobian_positive_valued_mean,
            labs_args=dict(
                title="Only-Positive-Mean Jacobians",
                subtitle=self.name,
                caption=description,
                x="",
                y="",
            ),
        )

    def _plot_hessian(self, hess: IncrementalSecondMoment, description: str) -> None:
        #
        # complete hessian
        #

        # variance
        variance_matrix: pd.DataFrame = pd.DataFrame(
            hess.compute(), columns=self.datasets.features, index=self.datasets.features
        )
        variance_matrix.to_csv(
            os.path.join(
                self.file_structure.feature_importance_hessian_variance, f"{self.name}_{description.lower()}.csv"
            )
        )
        heat_map(
            variance_matrix,
            path=self.file_structure.feature_importance_hessian_variance,
            name=f"{self.name}_{description.lower()}",
            title_args=dict(label="Variance of Hessians", subtitle=description),
        )

        # signed mean
        signed_mean_matrix: pd.DataFrame = pd.DataFrame(
            hess.signed_mean, columns=self.datasets.features, index=self.datasets.features
        )
        signed_mean_matrix.to_csv(
            os.path.join(
                self.file_structure.feature_importance_hessian_signed_mean, f"{self.name}_{description.lower()}.csv"
            )
        )
        heat_map(
            signed_mean_matrix,
            path=self.file_structure.feature_importance_hessian_signed_mean,
            name=f"{self.name}_{description.lower()}",
            title_args=dict(label="Signed Mean of Hessians", subtitle=description),
        )

        # only positive mean
        positive_valued_mean: pd.DataFrame = pd.DataFrame(
            hess.positive_valued_mean, columns=self.datasets.features, index=self.datasets.features
        )
        positive_valued_mean.to_csv(
            os.path.join(
                self.file_structure.feature_importance_hessian_positive_valued_mean,
                f"{self.name}_{description.lower()}.csv",
            )
        )
        heat_map(
            positive_valued_mean,
            path=self.file_structure.feature_importance_hessian_positive_valued_mean,
            name=f"{self.name}_{description.lower()}",
            title_args=dict(label="Only-Positive-Mean of Hessians", subtitle=description),
        )

        #
        # diagonals
        #

        shared_args: t.Dict[str, dict] = dict(
            aes_args=dict(x="value", y="variable"), name=f"{self.name}_{description.lower()}"
        )

        # variance
        diagonals_variance: pd.DataFrame = (
            pd.DataFrame(torch.diagonal(hess.compute()).view(1, -1), columns=self.datasets.features)
            .melt()
            .reset_index()
            .sort_values("value")
        )
        diagonals_variance.to_csv(
            os.path.join(
                self.file_structure.feature_importance_hessian_diagonals_variance,
                f"{self.name}_{description.lower()}.csv",
            )
        )
        bar_plot(
            diagonals_variance,
            **shared_args,
            path=self.file_structure.feature_importance_hessian_diagonals_variance,
            labs_args=dict(
                title="diag(Variance of Hessians)",
                subtitle=self.name,
                caption=description,
                x="",
                y="",
            ),
        )

        # signed mean
        diagonals_signed_mean: pd.DataFrame = (
            pd.DataFrame(torch.diagonal(hess.signed_mean).view(1, -1), columns=self.datasets.features)
            .melt()
            .reset_index()
            .sort_values("value")
        )
        diagonals_signed_mean.to_csv(
            os.path.join(
                self.file_structure.feature_importance_hessian_diagonals_signed_mean,
                f"{self.name}_{description.lower()}.csv",
            )
        )
        bar_plot(
            diagonals_signed_mean,
            **shared_args,
            path=self.file_structure.feature_importance_hessian_diagonals_signed_mean,
            labs_args=dict(
                title="diag(Signed Mean of Hessians)",
                subtitle=self.name,
                caption=description,
                x="",
                y="",
            ),
        )

        # only positive mean
        diagonals_positive_valued_mean: pd.DataFrame = (
            pd.DataFrame(torch.diagonal(hess.positive_valued_mean).view(1, -1), columns=self.datasets.features)
            .melt()
            .reset_index()
            .sort_values("value")
        )
        diagonals_positive_valued_mean.to_csv(
            os.path.join(
                self.file_structure.feature_importance_hessian_diagonals_positive_valued_mean,
                f"{self.name}_{description.lower()}.csv",
            )
        )
        bar_plot(
            diagonals_positive_valued_mean,
            **shared_args,
            path=self.file_structure.feature_importance_hessian_diagonals_positive_valued_mean,
            labs_args=dict(
                title="diag(Only-Positive-Mean of Hessians)",
                subtitle=self.name,
                caption=description,
                x="",
                y="",
            ),
        )
