import typing as t

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from attr import define
from gym.spaces import Discrete
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from typeguard import typechecked

from fin_dl import GOLDEN_RATION


class MDPConstants:
    LABEL_DATE: str = "Current Date"
    LABEL_STATES: str = "Current States"
    LABEL_POSITION: str = "Current Position"
    LABEL_NEXT_POSITION: str = "Next Position"
    LABEL_ACTION: str = "Action"
    LABEL_AVAILABLE_ACTIONS: str = "Actions Available"
    LABEL_ACTION_MASK: str = "Action Mask"
    LABEL_REALIZED_REWARD: str = "Realized Reward"
    LABEL_TRUE_REWARD: str = "True Reward"
    LABEL_REWARD: str = "Reward"
    LABEL_OPTIMAL_REWARD: str = "Optimal Reward"
    LABEL_NEXT_STATES: str = "Next States"
    LABEL_IS_TERMINAL_STATE: str = "Is current_date terminal"


class MDPTuple(MDPConstants):
    date: str
    states: pd.Series
    action: int
    realized_reward: float
    true_reward: float
    optimal_reward: float
    next_states: pd.Series
    is_state_terminal: t.Optional[bool]
    position: t.Optional[int]
    next_position: t.Optional[int]

    @typechecked
    def __init__(
        self,
        date: str,
        states: pd.Series,
        action: int,
        reward: float,
        true_reward: float,
        optimal_reward: float,
        next_states: pd.Series,
        is_state_terminal: t.Optional[bool] = None,
        position: t.Optional[int] = None,
        next_position: t.Optional[int] = None,
    ) -> None:
        self.date = date
        self.states = states
        self.action = action
        self.realized_reward = reward
        self.true_reward = true_reward
        self.optimal_reward = optimal_reward
        self.next_states = next_states
        self.is_state_terminal = is_state_terminal
        self.position = position
        self.next_position = next_position

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.LABEL_DATE}: {self.date}",
                f"{self.LABEL_STATES}: {self.states}",
                f"{self.LABEL_ACTION}: {Actions().to_string(action=self.action)}",
                f"{self.LABEL_REALIZED_REWARD}: {self.realized_reward}",
                f"{self.LABEL_TRUE_REWARD}: {self.true_reward}",
                f"{self.LABEL_OPTIMAL_REWARD}: {self.optimal_reward}",
                f"{self.LABEL_NEXT_STATES}: {self.next_states}",
                f"{self.LABEL_POSITION}: {Positions().to_string(position=self.position)}",
                f"{self.LABEL_NEXT_POSITION}: {Positions().to_string(position=self.next_position)}",
                f"{self.LABEL_IS_TERMINAL_STATE}: {self.is_state_terminal}",
            ]
        )

    def __repr__(self) -> str:
        return self.summary()

    @property
    def main_tuple(self) -> t.Tuple[pd.Series, int, float, t.Union[pd.Series, float]]:
        return self.states, self.action, self.realized_reward, self.next_states

    def gym_tuple(self, include_mask: bool = True) -> t.Tuple[np.ndarray, float, bool, t.Dict]:
        info: t.Dict[str, t.Any] = {
            self.LABEL_REWARD: self.realized_reward,
            self.LABEL_OPTIMAL_REWARD: self.optimal_reward,
            self.LABEL_TRUE_REWARD: self.true_reward,
        }
        return (
            self.states.values,
            self.realized_reward,
            self.is_state_terminal,
            dict(**info, **{self.LABEL_ACTION_MASK: Actions().action_mask(position=self.next_position)})
            if include_mask
            else info,
        )

    def multi_agent_tuple(self, agent_id: t.Union[str, int]) -> t.Tuple[t.Dict, ...]:
        return (
            {agent_id: self.states.values},
            {agent_id: self.realized_reward},
            {"__all__": self.is_state_terminal},
            {agent_id: {self.LABEL_ACTION_MASK: Actions().action_mask(position=self.next_position)}},
        )


class Positions:
    OUT: int = 0
    LONG: int = 1

    current: int

    LABEL_OUT: str = "Out"
    LABEL_LONG: str = "Long"

    def __init__(self) -> None:
        self.reset()

    def to_string(self, position: int) -> str:
        if position == self.OUT:
            return self.LABEL_OUT
        elif position == self.LONG:
            return self.LABEL_LONG

    def to_int(self, position: str) -> int:
        if position == self.LABEL_OUT:
            return self.OUT
        elif position == self.LABEL_LONG:
            return self.LONG

    def info(self) -> str:
        return "\n".join([f"{self.LABEL_OUT}: {self.OUT}", f"{self.LABEL_LONG}: {self.LONG}"])

    def update(self, action: int) -> None:
        self.current = self.new_position(action=action)

    def new_position(self, action: int) -> int:
        if action == Actions().SELL:
            return self.OUT
        elif action == Actions().BUY:
            return self.LONG
        elif action == Actions().NOTHING:
            return self.position

    def reset(self) -> None:
        self.current = self.OUT

    @property
    def position(self) -> int:
        return self.current

    @property
    def inverse(self) -> int:
        if self.current == self.OUT:
            return self.LONG
        else:
            return self.OUT


class PositionsMultiAgent(Positions):
    def __init__(self) -> None:
        super().__init__()

    def new_position(self, action: int) -> int:
        if action == ActionsMultiAgent().ACTIVE_ACTION:
            return self.inverse
        else:
            return self.position


class Actions:
    NOTHING: int = 0
    SELL: int = 1
    BUY: int = 2

    _LABEL_SELL: str = "Sell"
    _LABEL_NONE: str = "None"
    _LABEL_BUY: str = "Buy"

    def info(self) -> str:
        return "\n".join([f"Sell: {self.SELL}", f"Buy: {self.BUY}"])

    def action_to_sign(self, action: int) -> int:
        if action == self.SELL:
            return -1
        else:
            return 1

    def to_string(self, action: int, **kwargs) -> str:
        if action == self.SELL:
            return self._LABEL_SELL
        elif action == self.NOTHING:
            return self._LABEL_NONE
        elif action == self.BUY:
            return self._LABEL_BUY

    def __len__(self) -> int:
        return len([self.SELL, self.NOTHING, self.BUY])

    def actions_available(self, position: int) -> t.List[int]:
        if position == Positions().OUT:
            return [self.NOTHING, self.BUY]
        elif position == Positions().LONG:
            return [self.NOTHING, self.SELL]

    def action_mask(self, position: int) -> t.Dict[str, int]:
        if position == Positions().OUT:
            return {self._LABEL_NONE: True, self._LABEL_SELL: False, self._LABEL_BUY: True}
        elif position == Positions().LONG:
            return {self._LABEL_NONE: True, self._LABEL_SELL: True, self._LABEL_BUY: False}

    def get_action_space(self, position: int) -> Discrete:
        return Discrete(n=len(self.actions_available(position=position)))

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=len(self))

    @property
    def actions(self) -> t.Dict[str, int]:
        return {self._LABEL_NONE: self.NOTHING, self._LABEL_SELL: self.SELL, self._LABEL_BUY: self.BUY}


@define
class ActionsMultiAgent(Actions):
    LABEL_NONE: str = "None"
    LABEL_BUY: str = "Buy"
    LABEL_SELL: str = "Sell"

    PASSIVE_ACTION: int = 0
    ACTIVE_ACTION: int = 1

    LABEL_ACTIVE_ACTION: str = "Active"
    LABEL_PASSIVE_ACTION: str = "Passive"

    ACTIONS_STR_TO_INT: t.Dict[int, t.Dict[str, int]] = {
        Positions().OUT: {LABEL_NONE: PASSIVE_ACTION, LABEL_BUY: ACTIVE_ACTION},
        Positions().LONG: {LABEL_NONE: PASSIVE_ACTION, LABEL_SELL: ACTIVE_ACTION},
    }
    ACTIONS_INT_TO_STR: t.Dict[str, t.Dict[str, int]] = {
        Positions().LABEL_OUT: {PASSIVE_ACTION: LABEL_NONE, ACTIVE_ACTION: LABEL_BUY},
        Positions().LABEL_LONG: {PASSIVE_ACTION: LABEL_NONE, ACTIVE_ACTION: LABEL_SELL},
    }

    # noinspection PyMethodOverriding
    def action_to_sign(self, action: int, position: int) -> int:
        if position == Positions().LONG:
            if action == self.ACTIONS_STR_TO_INT[position][self.LABEL_SELL]:
                return -1
        else:
            return 1

    # noinspection PyMethodOverriding
    def to_string(self, action: int, position: int) -> str:
        if position == Positions().OUT:
            if action == self.PASSIVE_ACTION:
                return self.LABEL_NONE
            else:
                return self.LABEL_BUY
        elif position == Positions().LONG:
            if action == self.PASSIVE_ACTION:
                return self.LABEL_NONE
            else:
                return self.LABEL_SELL

    def get_action_space(self, position: t.Union[int, str]) -> Discrete:
        if isinstance(position, int):
            return Discrete(n=len(self.ACTIONS_STR_TO_INT[position]))
        elif isinstance(position, str):
            return Discrete(n=len(self.ACTIONS_INT_TO_STR[position]))

    @property
    def action_names(self) -> t.List[str]:
        return [self.LABEL_ACTIVE_ACTION, self.LABEL_PASSIVE_ACTION]

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=len(self.action_names))


class MDPHistory(MDPConstants):
    _dates: t.List[str]
    _positions: t.List[str]
    _realized_rewards: t.List[float]
    _true_rewards: t.List[float]
    _optimal_rewards: t.List[float]
    _actions: t.List[str]
    _states: t.List[t.List[float]]
    _next_states: t.List[t.List[float]]

    action: t.Union[Actions, ActionsMultiAgent]

    LABEL_AXIS_REWARD: str = "reward"
    LABEL_AXIS_ACTION: str = "action"
    LABEL_AXIS_POSITION: str = "position"
    LEGEND_LABEL_REALIZED_REWARD: str = "Realized financial return"
    LEGEND_LABEL_OPTIMAL_REWARD: str = "Optimal financial return"
    LEGEND_LABEL_ACTUAL_REWARD: str = "Actual financial return"

    def __init__(self, action: t.Union[Actions, ActionsMultiAgent]) -> None:
        self.action = action
        self._reset()

    def _reset(self) -> None:
        self._dates = []
        self._positions = []
        self._realized_rewards = []
        self._true_rewards = []
        self._optimal_rewards = []
        self._actions = []
        self._states = []
        self._next_states = []

    def update(
        self,
        date: str,
        position: int,
        realized_reward: float,
        true_reward: float,
        optimal_reward: float,
        action: int,
        states: t.List[float],
        next_state: t.List[float],
    ) -> None:
        self._dates.append(date)
        self._positions.append(Positions().to_string(position=position))
        self._realized_rewards.append(realized_reward)
        self._true_rewards.append(true_reward)
        self._optimal_rewards.append(optimal_reward)
        self._actions.append(self.action.to_string(action=action, position=position))
        self._states.append(states)
        self._next_states.append(next_state)

    def __getitem__(self, index: int) -> MDPTuple:
        return MDPTuple(
            date=self._dates[index],
            state=self._states[index],
            actions=self._actions[index],
            reward=self._actions[index],
            true_reward=self._true_rewards[index],
            optimal_reward=self._optimal_rewards[index],
            next_state=self._next_states[index],
            position=self._positions[index],
        )

    @property
    def data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.LABEL_DATE: pd.to_datetime(self._dates, infer_datetime_format=True),
                self.LABEL_STATES: self._states,
                self.LABEL_POSITION: self._positions,
                self.LABEL_ACTION: self._actions,
                self.LABEL_REALIZED_REWARD: self._realized_rewards,
                self.LABEL_OPTIMAL_REWARD: self._optimal_rewards,
                self.LABEL_TRUE_REWARD: self._true_rewards,
                self.LABEL_NEXT_STATES: self._next_states,
            }
        )

    @property
    def reward_cum_sum(self) -> pd.Series:
        return (
            self.data_frame.set_index(self.LABEL_DATE)
            .loc[:, [self.LABEL_TRUE_REWARD, self.LABEL_REALIZED_REWARD, self.LABEL_OPTIMAL_REWARD]]
            .sum()
        )

    def plot(self, filename: t.Optional[str] = None) -> None:
        fig: Figure
        axes: t.Dict[str, t.Any]
        frame: pd.DataFrame = self.data_frame
        color: np.ndarray = np.where(
            frame[self.LABEL_REALIZED_REWARD] >= frame[self.LABEL_OPTIMAL_REWARD], "#55a768", "#C44F51"
        )
        fig, axes = plt.subplot_mosaic(
            [
                [self.LABEL_AXIS_REWARD],
                [self.LABEL_AXIS_POSITION],
                [self.LABEL_AXIS_ACTION],
            ],
            gridspec_kw=dict(height_ratios=[1, 1 / GOLDEN_RATION, 1 / GOLDEN_RATION]),
            figsize=(np.max([len(frame) / np.log(len(frame) + 1), 5]), 5),
        )
        axes[self.LABEL_AXIS_REWARD].plot(
            pd.DatetimeIndex(frame[self.LABEL_DATE]),
            frame[self.LABEL_REALIZED_REWARD],
            label=self.LEGEND_LABEL_REALIZED_REWARD,
        )
        axes[self.LABEL_AXIS_REWARD].plot(
            pd.DatetimeIndex(frame[self.LABEL_DATE]),
            frame[self.LABEL_OPTIMAL_REWARD],
            linestyle="dotted",
            label=self.LEGEND_LABEL_OPTIMAL_REWARD,
            alpha=0.5,
        )
        axes[self.LABEL_AXIS_REWARD].plot(
            pd.DatetimeIndex(frame[self.LABEL_DATE]),
            frame[self.LABEL_TRUE_REWARD],
            linestyle="dashed",
            alpha=0.75,
            label=self.LEGEND_LABEL_ACTUAL_REWARD,
        )
        axes[self.LABEL_AXIS_REWARD].scatter(
            pd.DatetimeIndex(frame[self.LABEL_DATE]), frame[self.LABEL_REALIZED_REWARD], color=color, marker="o"
        )
        axes[self.LABEL_AXIS_REWARD].legend(
            loc="best",
            # borderaxespad=0.0,
            fontsize="x-small",
            # fancybox=True,
            # framealpha=0.0,
        )
        axes[self.LABEL_AXIS_REWARD].set_xticks([])
        for axis_name, frame_column in zip(
            [self.LABEL_AXIS_POSITION, self.LABEL_AXIS_ACTION], [self.LABEL_POSITION, self.LABEL_ACTION]
        ):
            axes[axis_name].plot(pd.DatetimeIndex(frame[self.LABEL_DATE]), frame[frame_column])
            axes[axis_name].scatter(
                pd.DatetimeIndex(frame[self.LABEL_DATE]), frame[frame_column], color=color, marker="o"
            )
        axes[self.LABEL_AXIS_POSITION].set_xticks([])
        axes[self.LABEL_AXIS_ACTION].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4)))
        axes[self.LABEL_AXIS_ACTION].xaxis.set_minor_locator(mdates.MonthLocator())
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
