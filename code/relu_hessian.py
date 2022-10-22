import typing as t

import torch
from torch import nn
from torch.autograd.functional import hessian, jacobian

torch.manual_seed(128)

inputs: torch.Tensor = torch.rand((5, 1, 10))
activation_function_name: str = "ReLU"

seq: nn.Sequential = nn.Sequential(
    nn.Linear(inputs.size(-1), 10),
    getattr(nn, activation_function_name)(),
    torch.nn.Linear(10, 5),
    getattr(nn, activation_function_name)(),
    nn.Linear(5, 1),
)

print(seq.forward(inputs))

print("All elements zero?: ", torch.all(jacobian(seq.forward, inputs) == 0.0))

container: t.List[torch.Tensor] = []
for x in inputs:
    hess: torch.Tensor = hessian(seq.forward, x)
    print(hess)
    # noinspection PyTypeChecker
    print(torch.all(hess == 0.0))
    container.append(hess)

# noinspection PyTypeChecker
print("All elements zero?: ", torch.all(torch.stack(container, dim=0) == 0.0))
