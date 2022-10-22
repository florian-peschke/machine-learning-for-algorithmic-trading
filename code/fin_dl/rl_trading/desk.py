import typing as t

import gym
import numpy as np
import pandas as pd
from numpy.random._generator import Generator
from typeguard import typechecked

from fin_dl.rl_trading.data import CustomIterator
from fin_dl.rl_trading.mdp import Actions, ActionsMultiAgent, MDPHistory, MDPTuple, Positions, PositionsMultiAgent
from fin_dl.rl_trading.reward import Reward, RewardMultiAgent
from fin_dl.utilities import get_values


class TradingDesk:
    raw_data: pd.DataFrame
    date_label: str
    start_date: t.Optional[str]
    target_value: str
    number_of_data_points: t.Optional[int]

    observation_space: gym.spaces.Box
    state_names: t.List[str]
    reward: t.Union[Reward, RewardMultiAgent]
    STATES_DTYPE: t.Any = float
    use_multiple_agents: bool

    _data_iterator_object: CustomIterator
    _data_interator: t.Iterator[pd.Series]
    position: t.Union[Positions, PositionsMultiAgent]
    _mdp_history: MDPHistory

    @typechecked
    def __init__(
        self,
        data: pd.DataFrame,
        total_buying_fee: float,
        total_selling_fee: float,
        state_names: t.List[str],
        use_multiple_agents: bool = True,
        start_date: t.Optional[str] = None,
        date_label: str = "date",
        target_value: str = "adjClose",
        number_of_data_points: t.Optional[int] = None,
    ) -> None:
        self.data = data
        self.date_label = date_label
        self.target_value = target_value
        self.state_names = state_names
        self.number_of_data_points = number_of_data_points
        self.use_multiple_agents = use_multiple_agents
        if self.use_multiple_agents:
            self.reward = RewardMultiAgent(total_buying_fee=total_buying_fee, total_selling_fee=total_selling_fee)
            self.position = PositionsMultiAgent()
        else:
            self.reward = Reward(total_buying_fee=total_buying_fee, total_selling_fee=total_selling_fee)
            self.position = Positions()

        self._set_start_date(start_date=start_date)
        self.reset()

    def _set_start_date(self, start_date: t.Optional[str] = None) -> None:
        if start_date is None:
            self.start_date = self.trading_days_available.min()
        else:
            self.start_date = start_date

    def _set_iterator(self) -> None:
        self._data_iterator_object = CustomIterator(
            data=self.data.loc[get_values(frame=self.data, label=self.date_label) >= self.start_date, :],
            date_label=self.date_label,
            number_of_data_points=self.number_of_data_points,
        )
        self._data_interator = iter(self._data_iterator_object)

    def step(self, action: t.Union[int, np.int64], reward_as_log: bool = True) -> MDPTuple:
        action: int = int(action)

        # get current and next trading inputs
        stock_data: pd.Series = next(self._data_interator)
        next_stock_data: pd.Series = self._data_iterator_object.get_next_values().astype(self.STATES_DTYPE)

        # get date
        date: str = self._data_iterator_object.current_date

        # infer states
        states: pd.Series = stock_data[self.state_names].astype(self.STATES_DTYPE)
        next_states: pd.Series = next_stock_data[self.state_names]

        # get current_position
        current_position: int = self.position.position

        # set target_label value
        current_target_value: float = float(stock_data[self.target_value])
        next_target_value: float = float(next_stock_data[self.target_value])

        # retrieve rewards
        true_reward: float
        optimal_reward: float
        realized_reward: float
        true_reward, optimal_reward, realized_reward = self.reward.get_reward(
            action=action,
            position=current_position,
            current_value=current_target_value,
            next_value=next_target_value,
            reward_as_log=reward_as_log,
        )

        # set bool
        is_state_terminal: bool = not self._data_iterator_object.has_next()

        # update current_position
        self.position.update(action=action)

        # update history
        self._mdp_history.update(
            date=date,
            position=current_position,
            realized_reward=realized_reward,
            true_reward=true_reward,
            optimal_reward=optimal_reward,
            action=action,
            states=states.tolist(),
            next_state=next_states.tolist(),
        )

        # return mdp tuple
        return MDPTuple(
            date=date,
            states=states,
            action=action,
            reward=realized_reward,
            true_reward=true_reward,
            optimal_reward=optimal_reward,
            next_states=next_states,
            is_state_terminal=is_state_terminal,
            position=current_position,
            next_position=self.position.position,
        )

    def reset(self, seed: t.Optional[int] = None) -> None:
        if seed is not None:
            self._set_start_date(
                start_date=str(np.random.default_rng(seed).choice(self.trading_days_available, size=1)[0])
            )
        else:
            self._set_start_date()
        self._set_iterator()
        self._mdp_history = MDPHistory(action=ActionsMultiAgent() if self.use_multiple_agents else Actions())
        self.position.reset()

    @property
    def trading_days_available(self) -> np.ndarray:
        return get_values(frame=self.data, label=self.date_label).sort_values().unique()

    @property
    def trading_days_used(self) -> np.ndarray:
        return self._data_iterator_object.dates

    @property
    def history(self) -> MDPHistory:
        return self._mdp_history

    @property
    def initial_states(self) -> np.ndarray:
        return self._data_iterator_object.initial_values[self.state_names].values.astype(self.STATES_DTYPE)
