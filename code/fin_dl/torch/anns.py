import re
import typing as t

import optuna
import torch
from optuna.trial import FrozenTrial
from torch import nn
from torch.nn.parameter import Parameter
from typeguard import typechecked

from fin_dl.torch import device
from fin_dl.torch.optuna_config import OptunaChoices
from fin_dl.torch.utils import are_grads_ok, init_parameter, init_parameters


class _ModuleWrapper(nn.Module):
    main_modules: nn.ModuleDict

    def __init__(self) -> None:
        super().__init__()

    def init_parameters(self) -> None:
        init_parameters(self.parameters())

    def are_grads_ok(self, verbose: bool = True) -> bool:
        return are_grads_ok(self.parameters(), verbose=verbose)

    @property
    def gradients(self) -> t.List[torch.Tensor]:
        return [parameter.grad for parameter in self.parameters()]

    @property
    def parameter_list(self) -> t.List[Parameter]:
        return list(self.parameters())


class CustomFNN(_ModuleWrapper):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        sequence_len: int,
        trial: t.Union[optuna.Trial, FrozenTrial],
        final_activation_name: t.Optional[str] = None,
        optuna_choices: OptunaChoices = OptunaChoices(),
        verbose: bool = False,
    ) -> None:
        super().__init__()

        # init module dict
        self.main_modules = nn.ModuleDict()

        # draw number of hidden stacks
        number_of_hidden_stacks: int = trial.suggest_int(
            name="number_of_hidden_stacks", low=optuna_choices.stack_depth.min, high=optuna_choices.stack_depth.max
        )

        # draw activation function name
        activation_function_name: str = trial.suggest_categorical(
            name=f"lin_act", choices=optuna_choices.ACTIVATION_FUNCTION_NAMES
        )

        # regularize?
        do_regularization: bool = trial.suggest_categorical("do_regularization", choices=[True, False])

        use_normalization_layer: bool = False
        use_1d: bool = False
        # if regularization is chosen
        if do_regularization:
            # use_normalization_layer | true | false => use dropout layer
            use_normalization_layer: bool = trial.suggest_categorical(
                name="use_normalization_layer", choices=[True, False]
            )
            if not use_normalization_layer:
                # whether to choose the Dropout1D
                use_1d = trial.suggest_categorical("use_dropout_1d", choices=[True, False])

        if verbose:
            print(
                re.sub(
                    "(\\n)+",
                    "\n",
                    "\n".join(
                        [
                            f"Regularize: {do_regularization}",
                            f"Use BatchNorm1d: {use_normalization_layer}" if do_regularization else "",
                            f"Use Dropout1d: {use_1d}" if do_regularization and not use_normalization_layer else "",
                            f"Activation function: {activation_function_name}",
                            f"Number of stacks: {number_of_hidden_stacks}",
                        ]
                    ),
                )
            )

        # update module dict
        for number in range(number_of_hidden_stacks):
            out_features: int = trial.suggest_categorical(name=f"out_lin{number}", choices=optuna_choices.units)

            # add linear layer
            self.main_modules.update(
                {
                    f"lin{number}": nn.Linear(in_features=input_size, out_features=out_features),
                }
            )

            # add activation function
            self.main_modules.update({f"lin_act{number}": getattr(nn, activation_function_name)()})

            if do_regularization:
                if use_normalization_layer:
                    # draw batch norm hyperparameters
                    eps: float = trial.suggest_uniform(f"eps_batch_norm{number}", low=1e-6, high=1e-4)
                    momentum = trial.suggest_uniform(f"momentum__batch_norma{number}", low=0.05, high=0.2)
                    # add batch normalization layer
                    self.main_modules.update(
                        {f"batch_norma{number}": nn.BatchNorm1d(sequence_len, eps=eps, momentum=momentum)}
                    )
                else:
                    # add a dropout layer for regularization purpose
                    if use_1d:
                        self.main_modules.update(
                            {
                                f"1d_dropout{number}": nn.Dropout1d(
                                    p=trial.suggest_categorical(
                                        name=f"1d_dropout_rate{number}", choices=optuna_choices.dropout_rates
                                    )
                                )
                            }
                        )
                    else:
                        self.main_modules.update(
                            {
                                f"dropout{number}": nn.Dropout(
                                    p=trial.suggest_categorical(
                                        name=f"dropout_rate{number}", choices=optuna_choices.dropout_rates
                                    )
                                )
                            }
                        )

            # update input_size
            input_size = out_features

        # flatten module preceding the final output module
        self.main_modules.update({"flatten": nn.Flatten()})

        # final output layer
        self.main_modules.update(
            {"fin_linear": nn.Linear(in_features=input_size * sequence_len, out_features=output_size)}
        )

        # add final activation function (optional)
        if final_activation_name is not None:
            self.main_modules.update({"fin_lin_act": getattr(nn, final_activation_name)()})

    def forward(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        for module_name, module in self.main_modules.items():
            try:
                batch = module(batch)
            except RuntimeError as e:
                raise RuntimeError(
                    "\n".join([f"module_name={module_name}", f"module={module}", f"input={batch}", str(e)])
                ) from e
        return batch


class CustomLSTM(_ModuleWrapper):
    output_modules: nn.Sequential

    _hidden_sizes: t.Dict[str, int]

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int,
        sequence_len: int,
        trial: t.Union[optuna.Trial, FrozenTrial],
        final_activation_name: t.Optional[str] = None,
        optuna_choices: OptunaChoices = OptunaChoices(),
        return_sequence: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        # init attributes
        self.main_modules = nn.ModuleDict()
        self._hidden_sizes = {}

        # draw number of hidden stacks
        number_of_hidden_stacks: int = trial.suggest_int(
            name="number_of_hidden_stacks", low=optuna_choices.stack_depth.min, high=optuna_choices.stack_depth.max
        )

        # regularize?
        do_regularization: bool = trial.suggest_categorical("do_regularization", choices=[True, False])

        use_normalization_layer: bool = False
        use_1d: bool = False
        # if regularization is chosen
        if do_regularization:
            # use_normalization_layer | true | false => use dropout layer
            use_normalization_layer: bool = trial.suggest_categorical(
                name="use_normalization_layer", choices=[True, False]
            )
            if not use_normalization_layer:
                # whether to choose the Dropout1D
                use_1d = trial.suggest_categorical("use_dropout_1d", choices=[True, False])

        if verbose:
            print(
                re.sub(
                    "(\\n)+",
                    "\n",
                    "\n".join(
                        [
                            f"Regularize: {do_regularization}",
                            f"Use BatchNorm1d: {use_normalization_layer}" if do_regularization else "",
                            f"Use Dropout1d: {use_1d}" if do_regularization and not use_normalization_layer else "",
                            f"Number of stacks: {number_of_hidden_stacks}",
                        ]
                    ),
                )
            )

        # create layers (modules)
        input_size: int = input_size
        hidden_size: int
        for number in range(number_of_hidden_stacks):
            # draw hidden size
            hidden_size: int = trial.suggest_categorical(
                name=f"hidden_size_lstm_cell{number}", choices=optuna_choices.units
            )

            # add lstm cell
            lstm_cell_name: str = f"lstm_cell{number}"
            self.main_modules.update(
                {
                    lstm_cell_name: nn.LSTMCell(input_size=input_size, hidden_size=hidden_size),
                }
            )

            # append input and hidden size the lists of the class
            self._hidden_sizes.update({lstm_cell_name: hidden_size})

            if do_regularization:
                if use_normalization_layer:
                    # draw batch norm hyperparameters
                    eps: float = trial.suggest_uniform(f"eps_batch_norm{number}", low=1e-6, high=1e-4)
                    momentum = trial.suggest_uniform(f"momentum__batch_norma{number}", low=0.05, high=0.2)
                    # add batch normalization layer
                    self.main_modules.update(
                        {f"batch_norma{number}": nn.BatchNorm1d(hidden_size, eps=eps, momentum=momentum)}
                    )
                else:
                    # add a dropout layer for regularization purpose
                    if use_1d:
                        self.main_modules.update(
                            {
                                f"1d_dropout{number}": nn.Dropout1d(
                                    p=trial.suggest_categorical(
                                        name=f"1d_dropout_rate{number}", choices=optuna_choices.dropout_rates
                                    )
                                )
                            }
                        )
                    else:
                        self.main_modules.update(
                            {
                                f"dropout{number}": nn.Dropout(
                                    p=trial.suggest_categorical(
                                        name=f"dropout_rate{number}", choices=optuna_choices.dropout_rates
                                    )
                                )
                            }
                        )

            # the hidden size of this lstm cell is the next lstm's input size
            input_size = hidden_size

        # init output sequential
        self.output_modules = nn.Sequential()

        if not return_sequence:
            # flatten
            self.output_modules.add_module(
                "fin_flatten",
                nn.Flatten(),
            )

        # final output layer
        self.output_modules.add_module(
            "fin_lin",
            nn.Linear(
                in_features=input_size if return_sequence else input_size * sequence_len, out_features=output_size
            ),
        )

        # add final activation function (optional)
        if final_activation_name is not None:
            self.output_modules.add_module(
                "fin_lin_act",
                getattr(nn, final_activation_name)(),
            )

    # @torchsnooper.snoop()
    def forward(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        """It is assumed that the tensor has the shape (batch, sequence_len, input_size)"""

        # infer batch size
        batch_size: int = batch.size(0)

        # init hidden state and hidden cell state tensors
        hidden_states: t.Dict[str, torch.Tensor] = self._get_initialized_hidden_states_tensor(batch_size)
        hidden_cell_states: t.Dict[str, torch.Tensor] = self._get_initialized_hidden_states_tensor(batch_size)

        # reshape batch to size=(sequence_len, batch_size, input_size)
        batch = batch.view(batch.size(1), batch_size, -1)

        # container
        output_container: t.List[torch.Tensor] = []

        # iterate over the first dimension
        for i in range(batch.size(0)):
            io: torch.Tensor = batch[i]
            # work through the module dictionary
            module_name: str
            module: nn
            for module_name, module in self.main_modules.items():
                if isinstance(module, nn.LSTMCell):
                    try:
                        hidden_states[module_name], hidden_cell_states[module_name] = module(
                            io, (hidden_states[module_name], hidden_cell_states[module_name])
                        )
                        # the input of the next layer/module is the hidden state of this lstm cell's output
                        io = hidden_states[module_name]
                    except RuntimeError as e:
                        raise RuntimeError(
                            "\n".join(
                                [
                                    f"module_name={module_name}",
                                    f"module: {str(module)}",
                                    f"io={io}",
                                    f"io.size()={io.size()}",
                                    f"hidden_states: {hidden_states[module_name]}",
                                    f"hidden_states.size(): {hidden_states[module_name].size()}",
                                    f"hidden_cell_states: {hidden_cell_states[module_name]}",
                                    f"hidden_cell_states.size(): {hidden_cell_states[module_name].size()}",
                                ]
                            )
                        ) from e
                else:
                    io = module(io)

            # append output to the container
            output_container.append(io)

        # stack 'output_container' to tensor
        io: torch.Tensor = torch.stack(output_container, dim=0)

        # reshape from size=(sequence_len, batch_size, input_size) to size=(batch_size, sequence_len, input_size)
        return self.output_modules(io.view(io.size(1), io.size(0), -1))

    def _get_initialized_hidden_states_tensor(self, batch_size: int) -> t.Dict[str, torch.Tensor]:
        """initialize state tensor by applying Xavier initialization"""
        return {
            name: self._get_initialized_state_tensors(batch_size=batch_size, hidden_size=hidden_size)
            for name, hidden_size in self._hidden_sizes.items()
        }

    @staticmethod
    def _get_initialized_state_tensors(batch_size: int, hidden_size: int) -> torch.Tensor:
        tensor: torch.Tensor = torch.empty(batch_size, hidden_size, device=device)
        return init_parameter(tensor, return_tensor=True)

    @property
    def hidden_sizes(self) -> t.Dict[str, int]:
        return self._hidden_sizes
