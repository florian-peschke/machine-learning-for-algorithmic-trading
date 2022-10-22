import os
import typing as t

import pandas as pd
from lets_plot import *
from lets_plot import aes, facet_grid, geom_line, geom_point, ggplot, ggsave, labs
from lets_plot.bistro.corr import *
from lets_plot.plot.core import PlotSpec

from fin_dl import GGPLOT_THEME


def bar_plot(
    frame: pd.DataFrame,
    aes_args: t.Dict[str, t.Any],
    size: t.Optional[t.Tuple[int, int]] = None,
    path: t.Optional[str] = None,
    name: t.Optional[str] = None,
    labs_args: t.Optional[t.Dict[str, t.Any]] = None,
    orientation: str = "y",
) -> t.Optional[PlotSpec]:
    if labs_args is None:
        labs_args = {}
    # create plot
    plot: PlotSpec = (
        ggplot(frame, aes(**aes_args))
        + geom_bar(stat="identity", orientation=orientation)
        + labs(**labs_args)
        + GGPLOT_THEME()
    )
    if size is not None:
        plot = plot + ggsize(*size)

    if path is None:
        return plot

    ggsave(plot, os.path.join(path, f"{name}.svg"))
    ggsave(plot, os.path.join(path, f"{name}.html"))


def heat_map(
    frame: pd.DataFrame,
    path: t.Optional[str] = None,
    name: t.Optional[str] = None,
    title_args: t.Optional[t.Dict[str, t.Any]] = None,
    tiles_and_points_args: t.Optional[t.Dict[str, t.Any]] = None,
    use_tiles: bool = True,
) -> t.Optional[PlotSpec]:
    if tiles_and_points_args is None:
        tiles_and_points_args = {}
    if title_args is None:
        title_args = {}

    # create plot
    plot: PlotSpec
    if use_tiles:
        plot = corr_plot(frame).tiles(**tiles_and_points_args).labels().build() + ggtitle(**title_args) + GGPLOT_THEME()
    else:
        plot = (
            corr_plot(frame).points(**tiles_and_points_args).labels().build() + ggtitle(**title_args) + GGPLOT_THEME()
        )

    if path is None:
        return plot

    ggsave(plot, os.path.join(path, f"{name}.svg"))
    ggsave(plot, os.path.join(path, f"{name}.html"))


def line_plot(
    frame: pd.DataFrame,
    aes_args: t.Dict[str, t.Any],
    size: t.Optional[t.Tuple[int, int]] = None,
    path: t.Optional[str] = None,
    name: t.Optional[str] = None,
    labs_args: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Optional[PlotSpec]:
    if labs_args is None:
        labs_args = {}
    # create plot
    plot: PlotSpec = ggplot(frame, aes(**aes_args)) + geom_line() + geom_point() + labs(**labs_args) + GGPLOT_THEME()

    if size is not None:
        plot = plot + ggsize(*size)

    if path is None:
        return plot

    ggsave(plot, os.path.join(path, f"{name}.svg"))
    ggsave(plot, os.path.join(path, f"{name}.html"))


def facet_line_plot(
    frame: pd.DataFrame,
    aes_args: t.Dict[str, t.Any],
    facet_args: t.Dict[str, t.Any],
    size: t.Optional[t.Tuple[int, int]] = None,
    path: t.Optional[str] = None,
    name: t.Optional[str] = None,
    labs_args: t.Optional[t.Dict[str, t.Any]] = None,
    geom_text_args: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Optional[PlotSpec]:
    if labs_args is None:
        labs_args = {}
    if geom_text_args is None:
        geom_text_args = {}
    # create plot
    plot: PlotSpec = (
        ggplot(frame, aes(**aes_args))
        + geom_line()
        + geom_point()
        + facet_grid(**facet_args)
        + geom_text(**geom_text_args)
        + labs(**labs_args)
        + GGPLOT_THEME()
    )
    if size is not None:
        plot = plot + ggsize(*size)

    if path is None:
        return plot

    ggsave(plot, os.path.join(path, f"{name}.svg"))
    ggsave(plot, os.path.join(path, f"{name}.html"))


def facet_bar_plot(
    frame: pd.DataFrame,
    aes_args: t.Dict[str, t.Any],
    facet_args: t.Dict[str, t.Any],
    size: t.Optional[t.Tuple[int, int]] = None,
    path: t.Optional[str] = None,
    name: t.Optional[str] = None,
    labs_args: t.Optional[t.Dict[str, t.Any]] = None,
    orientation: str = "y",
) -> t.Optional[PlotSpec]:
    if labs_args is None:
        labs_args = {}

    # create plot
    plot: PlotSpec = (
        ggplot(frame, aes(**aes_args))
        + geom_bar(stat="identity", orientation=orientation)
        + facet_grid(**facet_args)
        + labs(**labs_args)
        + GGPLOT_THEME()
    )

    if size is not None:
        plot = plot + ggsize(*size)

    if path is None:
        return plot

    ggsave(plot, os.path.join(path, f"{name}.svg"))
    ggsave(plot, os.path.join(path, f"{name}.html"))


def facet_violin_plot(
    frame: pd.DataFrame,
    aes_args: t.Dict[str, t.Any],
    violin_args: t.Dict[str, t.Any],
    facet_args: t.Dict[str, t.Any],
    size: t.Optional[t.Tuple[int, int]] = None,
    path: t.Optional[str] = None,
    name: t.Optional[str] = None,
    box_plot_width: int = 0.1,
    labs_args: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Optional[PlotSpec]:
    if labs_args is None:
        labs_args = {}

    # create plot
    plot: PlotSpec = (
        ggplot(frame, aes(**aes_args))
        + geom_violin(**violin_args)
        + geom_boxplot(width=box_plot_width)
        + facet_grid(**facet_args)
        + labs(**labs_args)
        + GGPLOT_THEME()
    )
    if size is not None:
        plot = plot + ggsize(*size)

    if path is None:
        return plot

    ggsave(plot, os.path.join(path, f"{name}.svg"))
    ggsave(plot, os.path.join(path, f"{name}.html"))


def plot_window_metrics(
    windows: pd.DataFrame,
    saving_path: t.Optional[str] = None,
    size: t.Optional[t.Tuple[int, int]] = None,
    name: t.Optional[str] = None,
) -> t.Optional[PlotSpec]:
    return facet_line_plot(
        windows.melt(ignore_index=False).reset_index(),
        aes_args=dict(x="index", y="value"),
        size=size,
        facet_args=dict(y="variable", scales="free"),
        labs_args=dict(title="Window Metrics", x="", y=""),
        path=saving_path,
        name=name,
    )
