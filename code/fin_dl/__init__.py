import typing as t
import warnings
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from lets_plot import theme_grey
from tqdm import tqdm


tqdm.pandas()
sns.set()
pd.set_option("use_inf_as_na", True)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
log_wandb: bool = True
wandb_project: str
if log_wandb:
    wandb_project = f"master_thesis ({datetime.now().strftime('%Y-%m-%d %H.%M')})"
else:
    wandb_project = "test"
SEED: int = 128
ENTITY: str = "flo0128"


def combinations(*args) -> t.List[t.Tuple]:
    return list(product(*args))


WEEKS_PER_YEAR: int = 52
GOLDEN_RATION: float = (1 + 5 ** (1 / 2)) / 2

GGPLOT_THEME: t.Callable = theme_grey
