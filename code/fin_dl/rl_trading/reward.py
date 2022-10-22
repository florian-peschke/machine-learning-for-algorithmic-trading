import typing as t
import warnings

import numpy as np
from typeguard import typechecked

from fin_dl.rl_trading.mdp import Actions, ActionsMultiAgent, Positions, PositionsMultiAgent


class Reward:
    total_buying_fee: float
    total_selling_fee: float

    def __init__(self, total_buying_fee: float, total_selling_fee: float) -> None:
        self.total_buying_fee = total_buying_fee
        self.total_selling_fee = total_selling_fee

    @staticmethod
    def get_change(current_value: float, next_value: float, reward_as_log: bool) -> float:
        try:
            if reward_as_log:
                return np.log((next_value / current_value))
            else:
                return (next_value - current_value) / current_value
        except ZeroDivisionError:
            print(dict(current_value=current_value, next_value=next_value, reward_as_log=reward_as_log))

    @staticmethod
    def is_transaction_necessary(position: int, action: int) -> bool:
        """
        Transaction is only necessary if the position will change after taking the respective action
        """
        if (
            (position == Positions().OUT and action == Actions().SELL)
            or (position == Positions().LONG and action == Actions().BUY)
            or action == Actions().NOTHING
        ):
            return False
        else:
            return True

    def get_adjusted_value(self, action: int, position: int, value: float) -> float:
        # adjust for transaction costs
        if self.is_transaction_necessary(position=position, action=action):
            if action == Actions().SELL:
                # selling implicitly lowers the stock price by the fee
                return value * np.subtract(1, self.total_selling_fee)
            elif action == Actions().BUY:
                # buying implicitly increases the stock price by the fee
                return value * np.add(1, self.total_buying_fee)
        return value

    def get_realized_reward(
        self, action: int, position: int, current_value: float, next_value: float, reward_as_log: bool = False
    ) -> float:
        if action == Actions().NOTHING:
            # the reward is the financial return ...
            true_reward: float = self.get_change(
                current_value=current_value, next_value=next_value, reward_as_log=reward_as_log
            )
            # ... that has to be signed depending on the position the agent is currently in
            if position == Positions().OUT:
                return -true_reward
            elif position == Positions().LONG:
                return true_reward
        else:
            # otherwise one has to adjust for transaction costs (fees)
            adjusted_current_value: float = self.get_adjusted_value(
                action=action, position=position, value=current_value
            )
            realized_reward: float = self.get_change(
                current_value=adjusted_current_value, next_value=next_value, reward_as_log=reward_as_log
            )
            return Actions().action_to_sign(action=action) * realized_reward

    def get_optimal_reward(
        self, position: int, current_value: float, next_value: float, reward_as_log: bool = False
    ) -> float:
        # the optimal reward is always positive
        true_reward: float = self.get_change(
            current_value=current_value, next_value=next_value, reward_as_log=reward_as_log
        )
        optimal_action: int
        if true_reward >= 0:
            optimal_action = Actions().BUY
        else:
            optimal_action = Actions().SELL
        realized_reward: float = self.get_realized_reward(
            action=optimal_action,
            position=position,
            current_value=current_value,
            next_value=next_value,
            reward_as_log=reward_as_log,
        )
        if true_reward >= 0:
            if position == Positions().OUT:
                # The realized reward can be negative due to transaction costs
                if realized_reward < 0:
                    # So the reward is the opportunity cost of not having bought the stock
                    return np.abs(realized_reward)
                # Otherwise it is the financial return received after buying the stock
                return realized_reward
            elif position == Positions().LONG:
                # Just get the true reward without adjusting for transaction costs
                return Actions().action_to_sign(action=optimal_action) * true_reward
        else:
            if position == Positions().OUT:
                # Just get the positive true reward without adjusting for transaction costs
                return Actions().action_to_sign(action=optimal_action) * true_reward
            elif position == Positions().LONG:
                # The realized reward can be negative due to transaction costs
                if realized_reward < 0:
                    # So the reward is the opportunity cost of not having bought the stock
                    return np.abs(realized_reward)
                # Otherwise it is the financial return received after buying the stock
                return realized_reward

    def get_reward(
        self, action: int, position: int, current_value: float, next_value: float, reward_as_log: bool = False
    ) -> t.Tuple[float, float, float]:
        true_reward: float = self.get_change(
            current_value=current_value, next_value=next_value, reward_as_log=reward_as_log
        )
        optimal_reward: float = self.get_optimal_reward(
            position=position,
            current_value=current_value,
            next_value=next_value,
            reward_as_log=reward_as_log,
        )
        realized_reward: float = self.get_realized_reward(
            action=action,
            position=position,
            current_value=current_value,
            next_value=next_value,
            reward_as_log=reward_as_log,
        )
        if realized_reward > optimal_reward:
            warnings.warn(
                self.get_info(
                    action=action,
                    position=position,
                    current_value=current_value,
                    next_value=next_value,
                    realized_reward=realized_reward,
                    true_reward=true_reward,
                    optimal_reward=optimal_reward,
                )
            )
        return true_reward, optimal_reward, realized_reward

    def get_info(
        self,
        action: int,
        position: int,
        current_value: float,
        next_value: float,
        realized_reward: float,
        true_reward: float,
        optimal_reward: float,
    ) -> str:
        return "\n".join(
            [
                f"realized_reward ({realized_reward}) > optimal_reward ({optimal_reward})",
                f"action={Actions().to_string(action=action)}",
                f"position={Positions().to_string(position=position)}",
                f"current_value={current_value}",
                f"next_value={next_value}",
                f"adjusted_values=" f"{self.get_adjusted_value(action=action, position=position, value=current_value)}",
                f"optimal_reward={optimal_reward}",
                f"realized_reward={realized_reward}",
                f"true_reward={true_reward}",
            ]
        )


# @define
class RewardMultiAgent(Reward):
    @typechecked
    def __init__(self, total_buying_fee: float, total_selling_fee: float) -> None:
        super().__init__(total_buying_fee=total_buying_fee, total_selling_fee=total_selling_fee)

    def get_adjusted_value(self, action: int, position: int, value: float) -> float:
        if action == ActionsMultiAgent().ACTIVE_ACTION:
            if position == PositionsMultiAgent().LONG:
                return value * np.subtract(1, self.total_selling_fee)
            elif action == PositionsMultiAgent().OUT:
                return value * np.add(1, self.total_buying_fee)
        return value

    def get_realized_reward(
        self, action: int, position: int, current_value: float, next_value: float, reward_as_log: bool = False
    ) -> float:
        if action == ActionsMultiAgent().PASSIVE_ACTION:
            true_reward: float = self.get_change(
                current_value=current_value, next_value=next_value, reward_as_log=reward_as_log
            )
            if position == PositionsMultiAgent().OUT:
                return -true_reward
            elif position == PositionsMultiAgent().LONG:
                return true_reward
        else:
            adjusted_current_value: float = self.get_adjusted_value(
                action=action, position=position, value=current_value
            )
            realized_reward: float = self.get_change(
                current_value=adjusted_current_value, next_value=next_value, reward_as_log=reward_as_log
            )
            return ActionsMultiAgent().action_to_sign(action=action, position=position) * realized_reward

    def get_optimal_reward(
        self, position: int, current_value: float, next_value: float, reward_as_log: bool = False
    ) -> float:
        true_reward: float = self.get_change(
            current_value=current_value, next_value=next_value, reward_as_log=reward_as_log
        )
        realized_reward: float = self.get_realized_reward(
            action=ActionsMultiAgent().ACTIVE_ACTION,
            position=position,
            current_value=current_value,
            next_value=next_value,
            reward_as_log=reward_as_log,
        )
        if true_reward >= 0:
            if position == Positions().OUT:
                # The realized reward can be negative due to transaction costs
                if realized_reward < 0:
                    # So the reward is the opportunity cost of not having bought the stock
                    return np.abs(realized_reward)
                # Otherwise it is the financial return received after buying the stock
                return realized_reward
            elif position == Positions().LONG:
                # Just get the true reward without adjusting for transaction costs
                return np.abs(true_reward)
        else:
            if position == Positions().OUT:
                # Just get the positive true reward without adjusting for transaction costs
                return np.abs(true_reward)
            elif position == Positions().LONG:
                # The realized reward can be negative due to transaction costs
                if realized_reward < 0:
                    # So the reward is the opportunity cost of not having bought the stock
                    return np.abs(realized_reward)
                # Otherwise it is the financial return received after buying the stock
                return realized_reward

    def get_info(
        self,
        action: int,
        position: int,
        current_value: float,
        next_value: float,
        realized_reward: float,
        true_reward: float,
        optimal_reward: float,
    ) -> str:
        return "\n".join(
            [
                f"realized_reward ({realized_reward}) > optimal_reward ({optimal_reward})",
                f"action={ActionsMultiAgent().to_string(action=action, position=position)}",
                f"position={Positions().to_string(position=position)}",
                f"current_value={current_value}",
                f"next_value={next_value}",
                f"adjusted_values={self.get_adjusted_value(action=action, position=position, value=current_value)}",
                f"optimal_reward={optimal_reward}",
                f"realized_reward={realized_reward}",
                f"true_reward={true_reward}",
            ]
        )
