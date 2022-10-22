import typing as t

import pandas as pd
import ray
from attr import define, field
from ray.rllib.agents import dqn
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from tqdm import tqdm
from typeguard import typechecked

from fin_dl.rl_trading.environment import StockExchange, StockExchangeMultiAgent
from fin_dl.rl_trading.mdp import PositionsMultiAgent
from fin_dl.rl_trading.performance import RLPostTraining, RLPostTrainingMultiAgent


@define
class RLStudy:
    symbol: str
    target_value: str
    training_episodes: int
    seed: t.Optional[int]
    total_buying_fee: int
    total_selling_fee: int
    trainer: t.Callable
    additional_config: dict

    training_data: pd.DataFrame
    test_data: pd.DataFrame
    transform_columns: t.List[str]

    single_agent_results: t.List[dict] = field(init=False)
    single_agent_train_cum_sum: t.List[pd.DataFrame] = field(init=False)
    multi_agent_results: t.List[dict] = field(init=False)
    multi_agent_train_cum_sum: t.List[pd.DataFrame] = field(init=False)

    _multi_agent: ray.rllib.agents = field(init=False)
    _single_agent: ray.rllib.agents = field(init=False)

    @typechecked
    def __init__(
        self,
        symbol: str,
        training_data: pd.DataFrame,
        test_data: pd.DataFrame,
        transform_columns: t.List[str],
        trainer: t.Callable = dqn.DQNTrainer,
        additional_config: dict = None,
        seed: t.Optional[int] = None,
        target_value: str = "close",
        training_episodes: int = 100,
        total_buying_fee=0.02,
        total_selling_fee=0.01,
    ) -> None:
        if additional_config is None:
            additional_config = {
                "framework": "torch",
                "horizon": None,
            }
        self.__attrs_init__(
            symbol=symbol,
            target_value=target_value,
            training_episodes=training_episodes,
            seed=seed,
            total_buying_fee=total_buying_fee,
            total_selling_fee=total_selling_fee,
            trainer=trainer,
            training_data=training_data,
            test_data=test_data,
            transform_columns=transform_columns,
            additional_config=additional_config,
        )

    def run_single_agent(self, return_eval: bool = False) -> t.Optional[RLPostTraining]:
        self._set_single_agent()
        self.single_agent_results = []
        self.single_agent_train_cum_sum = []
        for i in tqdm(range(self.training_episodes), desc="Training"):
            result: t.Dict[str, t.Any] = self._single_agent.train()
            self.single_agent_results.append(result)
            self.single_agent_train_cum_sum.append(
                pd.Series(self._single_agent.workers.local_worker().env.trading_desk.history.reward_cum_sum, name=i + 1)
                .to_frame()
                .T
            )
            print(pretty_print(result))
        return self._evaluate_single(return_self=return_eval)

    def _set_single_agent(self) -> None:
        self._single_agent = self.trainer(
            env=StockExchange,
            config={
                "env_config": dict(
                    trading_desk=self._trading_desk_arguments(use_multiple_agents=False, use_testing_data=False),
                    seed=None,
                ),
                **self.additional_config,
            },
        )

    def _evaluate_single(self, return_self: bool = False) -> t.Optional[RLPostTraining]:
        stock_exchange_testing: StockExchange = StockExchange(
            config=dict(
                trading_desk=self._trading_desk_arguments(use_multiple_agents=False, use_testing_data=True),
                seed=None,
            )
        )
        evaluation: RLPostTraining = RLPostTraining(agent=self._single_agent, environment=stock_exchange_testing)
        print(evaluation.rewards.sum())
        return evaluation if return_self else None

    def run_multi_agent(self, return_eval: bool = False) -> t.Optional[RLPostTrainingMultiAgent]:
        self._set_multi_agent()
        self.multi_agent_results = []
        self.multi_agent_train_cum_sum = []
        for i in tqdm(range(self.training_episodes), desc="Training"):
            result: t.Dict[str, t.Any] = self._multi_agent.train()
            self.multi_agent_results.append(result)
            self.multi_agent_train_cum_sum.append(
                pd.Series(self._multi_agent.workers.local_worker().env.trading_desk.history.reward_cum_sum, name=i + 1)
                .to_frame()
                .T
            )
            print(pretty_print(result))
        return self._evaluate_multi(return_self=return_eval)

    def _evaluate_multi(self, return_self: bool = False) -> t.Optional[RLPostTrainingMultiAgent]:
        stock_exchange_testing: StockExchangeMultiAgent = StockExchangeMultiAgent(
            config=dict(
                trading_desk=self._trading_desk_arguments(use_multiple_agents=True, use_testing_data=True),
                seed=self.seed,
            )
        )
        evaluation: RLPostTrainingMultiAgent = RLPostTrainingMultiAgent(
            agent=self._multi_agent, environment=stock_exchange_testing
        )
        print(evaluation.rewards.sum())
        return evaluation if return_self else None

    def _set_multi_agent(self) -> None:
        self._multi_agent = self.trainer(
            env=StockExchangeMultiAgent,
            config={
                "env_config": dict(
                    trading_desk=self._trading_desk_arguments(use_multiple_agents=True, use_testing_data=False),
                    seed=self.seed,
                ),
                **self.additional_config,
                "multiagent": {
                    "policies": {
                        PositionsMultiAgent().LABEL_LONG: PolicySpec(
                            policy_class=None,
                            observation_space=None,
                            action_space=None,
                            config={"gamma": 0.85},
                        ),
                        PositionsMultiAgent().LABEL_OUT: PolicySpec(
                            policy_class=None,
                            observation_space=None,
                            action_space=None,
                            config={"gamma": 0.85},
                        ),
                    },
                    "policies_to_train": [PositionsMultiAgent().LABEL_LONG, PositionsMultiAgent().LABEL_OUT],
                    "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: agent_id,
                },
            },
        )

    def _trading_desk_arguments(
        self, use_multiple_agents: bool = False, use_testing_data: bool = False
    ) -> t.Dict[str, t.Any]:
        return dict(
            data=self.test_data if use_testing_data else self.training_data,
            total_buying_fee=self.total_buying_fee,
            total_selling_fee=self.total_selling_fee,
            state_names=self.transform_columns,
            target_value=self.target_value,
            use_multiple_agents=use_multiple_agents,
        )

    @property
    def cum_sum_single(self) -> pd.DataFrame:
        return pd.concat(self.single_agent_train_cum_sum)

    @property
    def cum_sum_multi(self) -> pd.DataFrame:
        return pd.concat(self.multi_agent_train_cum_sum)
