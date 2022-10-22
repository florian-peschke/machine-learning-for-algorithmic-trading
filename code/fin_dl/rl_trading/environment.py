import typing as t
from abc import ABC
from typing import Tuple

import gym
import numpy as np
import ray
from attr import field
from gym.spaces import Discrete
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from typeguard import typechecked

from fin_dl.rl_trading.desk import TradingDesk
from fin_dl.rl_trading.mdp import Actions, ActionsMultiAgent, MDPConstants, MDPTuple, Positions


class StockExchange(gym.Env, MDPConstants):
    seed: t.Optional[int]
    trading_desk: TradingDesk

    observation_space: gym.spaces.Box
    action_space: gym.spaces.Discrete

    metadata: t.Dict[str, t.List[str]] = {"render.modes": ["human"]}

    @typechecked
    def __init__(self, config: t.Dict[str, t.Any]) -> None:
        """
        Initialize object.
        'config' argument requires two keys: ['trading_desk', 'seed'] with types [t.Dict[str, any], int]
        """
        self.trading_desk = TradingDesk(**config["trading_desk"])
        self.trading_desk.reset()
        self.seed = config["seed"]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.trading_desk.state_names),),
            dtype=float,
        )
        self.action_space = Actions().action_space

    def step(self, action: int) -> t.Tuple[np.ndarray, float, bool, t.Dict]:
        mdp_tuple: MDPTuple = self.trading_desk.step(action=action, reward_as_log=True)
        return mdp_tuple.gym_tuple()

    def reset(
        self, seed: t.Optional[int] = None, return_info: bool = False
    ) -> t.Union[np.ndarray, t.Tuple[np.ndarray, t.Dict]]:
        self.trading_desk.reset(seed=seed)
        if return_info:
            return (
                self.trading_desk.initial_states,
                {self.LABEL_ACTION_MASK: Actions().action_mask(position=Positions().OUT)},
            )
        else:
            return self.trading_desk.initial_states

    def render(self, mode: str = "human"):
        self.trading_desk.history.plot()

    def seed(self, seed: t.Optional[int] = None):
        return super().seed(seed)


class StockExchangeMultiAgent(MultiAgentEnv):
    seed: t.Optional[int]
    trading_desk: TradingDesk
    config: t.Dict[str, t.Any]

    observation_space: gym.spaces.Box = field(init=False)
    action_space: Discrete = field(init=False)
    action_spaces: t.Dict[str, Discrete] = field(init=False)

    agents: MultiAgentDict = {Positions().LABEL_OUT: 0, Positions().LABEL_LONG: 1}
    _agent_ids: t.Set = {Positions().LABEL_OUT, Positions().LABEL_LONG}

    metadata: t.Dict[str, t.List[str]] = {"render.modes": ["human"]}

    @typechecked
    def __init__(self, config: t.Dict[str, t.Any]) -> None:
        self.seed = config["seed"]
        self.trading_desk = TradingDesk(**config["trading_desk"])
        self.config = config
        self._post_init__()
        super().__init__()

    def _post_init__(self) -> None:
        self.trading_desk.reset()
        self.action_spaces = {
            agent_name: ActionsMultiAgent().get_action_space(position=agent_name) for agent_name in self.agents.keys()
        }
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.trading_desk.state_names),),
            dtype=float,
        )
        self.action_space = ActionsMultiAgent().action_space

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        print(action_dict)
        # get variables
        agent_name: str
        action: int
        print(action_dict)
        agent_name, action = list(action_dict.keys())[0], int(list(action_dict.values())[0])

        # get the mdp tuple
        mdp_tuple: MDPTuple = self.trading_desk.step(action=action, reward_as_log=True)

        # derive the new agent that depends on the new position
        new_agent: str = Positions().to_string(position=mdp_tuple.next_position)
        # noinspection PyTypeChecker
        return mdp_tuple.multi_agent_tuple(agent_id=new_agent)

    def reset(self, seed: t.Optional[int] = None) -> t.Dict[str, np.ndarray]:
        self.trading_desk.reset(seed=seed)
        return {Positions().LABEL_OUT: self.trading_desk.initial_states}
