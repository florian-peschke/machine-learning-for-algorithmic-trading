import os
import typing as t

import numpy as np
import pandas as pd
import ray
import tabulate
from attr import define, field
from lets_plot import *
from lets_plot.plot.core import PlotSpec
from ray.rllib import MultiAgentEnv
from ray.rllib.agents import ppo
from scipy.constants import golden_ratio
from typeguard import typechecked

from fin_dl import GGPLOT_THEME
from fin_dl.rl_trading.environment import StockExchange, StockExchangeMultiAgent
from fin_dl.rl_trading.mdp import MDPConstants

LetsPlot.setup_html()


@define
class RLPostTraining(MDPConstants):
    agent: ray.rllib.agents
    environment: t.Union[StockExchange, StockExchangeMultiAgent]
    frame: pd.DataFrame = field(init=False)

    @typechecked
    def __init__(self, agent: ray.rllib.agents, environment: t.Union[StockExchange, StockExchangeMultiAgent]) -> None:
        self.__attrs_init__(agent=agent, environment=environment)

    def __attrs_post_init__(self) -> None:
        self.frame = self._evaluate()

    def _evaluate(self) -> pd.DataFrame:
        observations: np.ndarray = self.environment.reset()
        is_state_terminal: bool = False
        while not is_state_terminal:
            action: int = self.agent.compute_single_action(observations, explore=False)
            reward: float
            info: t.Dict[str, t.Any]
            observations, reward, is_state_terminal, info = self.environment.step(action=action)
        return self.environment.trading_desk.history.data_frame

    def comparison_plot(
        self,
        sub_plot_weight: int = 500,
        target_label: str = "logR(close)",
        path: t.Optional[str] = None,
        name: str = "comparison_plot",
    ) -> t.Optional[GGBunch]:
        bunch: GGBunch = GGBunch()
        realized_label: str = "Realized"
        reward_columns: t.Dict[str, str] = {
            self.LABEL_TRUE_REWARD: "True",
            self.LABEL_OPTIMAL_REWARD: "Optimal",
        }
        counter: int = 0
        for column_name, description in reward_columns.items():
            bunch.add_plot(
                ggplot(self.frame, aes(x=self.LABEL_REALIZED_REWARD, y=column_name))
                + geom_point()
                + geom_abline(slope=1)
                + theme(legend_position="none", panel_grid="blank")
                + labs(
                    **dict(
                        title="RL Reward Comparison" if counter == 0 else "",
                        subtitle=f"{realized_label} vs. {description} {self.LABEL_REWARD.lower()}",
                        caption=f"{self.LABEL_REWARD.lower()} := financial  {target_label}",
                        x=f"{realized_label} {self.LABEL_REWARD.lower()}",
                        y=f"{description} {self.LABEL_REWARD.lower()}",
                    )
                )
                + GGPLOT_THEME(),
                sub_plot_weight * counter,
                0,
                sub_plot_weight,
                sub_plot_weight,
            )
            counter += 1
        if path is None:
            return bunch
        else:
            ggsave(bunch, os.path.join(path, f"{name}.svg"))
            ggsave(bunch, os.path.join(path, f"{name}.html"))

    def cum_sum_plot(
        self,
        sub_plot_weight: t.Optional[int] = None,
        subplot_height: t.Optional[int] = None,
        target_label: str = "logR(close)",
        format: str = "%b %Y",
        path: t.Optional[str] = None,
        name: str = "sum_sum_plot",
    ) -> t.Optional[PlotSpec]:
        if sub_plot_weight is not None and subplot_height is None:
            subplot_height = sub_plot_weight / golden_ratio
        g: PlotSpec = (
            ggplot(
                pd.melt(self.rewards.cumsum(), ignore_index=False, var_name="Type").reset_index(),
                aes(x=self.LABEL_DATE, y="value", group="Type"),
            )
            + geom_line(aes(color="Type"))
            + scale_x_datetime(format=format)
            + labs(
                **dict(
                    title="CumSum of Rewards",
                    subtitle="",
                    caption="",
                    x="",
                    y=f"CumsSum({target_label})",
                )
            )
            + GGPLOT_THEME()
        )
        if sub_plot_weight is not None:
            g = g + ggsize(sub_plot_weight, subplot_height)

        if path is None:
            return g
        else:
            self.rewards.cumsum().to_csv(os.path.join(path, f"{name}.csv"))
            ggsave(g, os.path.join(path, f"{name}.svg"))
            ggsave(g, os.path.join(path, f"{name}.html"))

    def detailed_plot(
        self,
        sub_plot_weight: int = 1000,
        subplot_height: t.Optional[int] = None,
        target_label: str = "logR(close)",
        format: str = "%b %Y",
        path: t.Optional[str] = None,
        name: str = "detailed_plot",
    ) -> t.Optional[GGBunch]:
        if subplot_height is None:
            subplot_height = sub_plot_weight / golden_ratio
        bunch: GGBunch = GGBunch()
        p: PlotSpec = ggplot(self.frame)
        # noinspection PyTypeChecker
        color: t.List[str] = (
            np.where(self.frame[self.LABEL_REALIZED_REWARD] >= self.frame[self.LABEL_OPTIMAL_REWARD], "green", "red")
            .astype(str)
            .tolist()
        )
        reward_columns: t.Dict[str, str] = {
            self.LABEL_TRUE_REWARD: "True",
            self.LABEL_OPTIMAL_REWARD: "Optimal",
            self.LABEL_REALIZED_REWARD: "Realized",
        }
        misc_columns: t.Dict[str, str] = {self.LABEL_POSITION: "Positions", self.LABEL_ACTION: "Actions"}
        counter: int = 0
        for column_name, description in reward_columns.items():
            bunch.add_plot(
                p
                + geom_line(aes(x=self.LABEL_DATE, y=column_name))
                + scale_x_datetime(format=format)
                + labs(
                    **dict(
                        title=f"RL Rewards := Financial {target_label}" if counter == 0 else "",
                        subtitle=f"{description} {self.LABEL_REWARD.lower()}",
                        caption=f"{description} {target_label}",
                        x="",
                        y="",
                    )
                )
                + theme(legend_position="bottom", panel_grid="blank")
                + GGPLOT_THEME(),
                0,
                subplot_height * counter,
                sub_plot_weight,
                subplot_height,
            )
            counter += 1
        for column_name, description in misc_columns.items():
            bunch.add_plot(
                p
                + geom_line(aes(x=self.LABEL_DATE, y=column_name))
                + scale_x_datetime(format=format)
                + geom_point(aes(x=self.LABEL_DATE, y=column_name, color=color))
                + theme(legend_position="none", panel_grid="blank")
                + labs(
                    **dict(
                        title="",
                        subtitle=description,
                        caption="red/orange: realized reward is lower than the optimal reward",
                        x="",
                        y="",
                    )
                )
                + GGPLOT_THEME()
                + theme(legend_position="none"),
                0,
                subplot_height * counter,
                sub_plot_weight,
                subplot_height,
            )
            counter += 1
        if path is None:
            return bunch
        else:
            ggsave(bunch, os.path.join(path, f"{name}.svg"))
            ggsave(bunch, os.path.join(path, f"{name}.html"))

    @property
    def rewards(self) -> pd.Series:
        return self.as_frame.set_index(self.LABEL_DATE).loc[
            :, [self.LABEL_TRUE_REWARD, self.LABEL_REALIZED_REWARD, self.LABEL_OPTIMAL_REWARD]
        ]

    def save_reward_sum(self, path: str, name: str = "single_agent_cum_sum") -> None:
        with open(os.path.join(path, f"{name}.txt"), "w") as f:
            print(
                tabulate.tabulate(self.rewards.sum().to_frame().T, headers="keys", showindex=False, tablefmt="latex"),
                file=f,
            )

    @property
    def as_frame(self) -> pd.DataFrame:
        try:
            return self.frame
        except AttributeError:
            return self._evaluate()


class RLPostTrainingMultiAgent(RLPostTraining):
    @typechecked
    def __init__(self, agent: ray.rllib.agents, environment: StockExchangeMultiAgent) -> None:
        super().__init__(agent=agent, environment=environment)

    def _evaluate(self) -> pd.DataFrame:
        observations: t.Dict[str, np.ndarray] = self.environment.reset()
        is_state_terminal: bool = False
        while not is_state_terminal:
            policy: str = list(observations.keys())[0]
            action: int = self.agent.compute_single_action(
                observations[policy], explore=False, policy_id=list(observations.keys())[0]
            )
            reward: float
            info: t.Dict[str, t.Dict[str, t.Any]]
            observations, reward, done, info = self.environment.step(action_dict={policy: action})
            is_state_terminal = done["__all__"]
        return self.environment.trading_desk.history.data_frame

    def save_reward_sum(self, path: str, name: str = "multi_agent_cum_sum") -> None:
        super().save_reward_sum(path=path, name=name)
