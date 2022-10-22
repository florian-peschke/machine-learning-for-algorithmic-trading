import numpy as np
import torch
from os import cpu_count

from fin_dl import SEED


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

NUMBER_OF_WORKERS: int = cpu_count()
TENSOR_DTYPE: str = "float"

print(f"Is Cuda available: {torch.cuda.is_available()}")
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
