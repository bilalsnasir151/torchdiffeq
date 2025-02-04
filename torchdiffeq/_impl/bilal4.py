import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


#iserles and norsett 1990, L stable 
_BILAL_4_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2, 2 / 3, 1 / 2, 1 / 3], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 2], dtype=torch.float64),
        torch.tensor([0, 2 / 3], dtype=torch.float64),
        torch.tensor([-5 / 2, 5 / 2, 1 / 2], dtype=torch.float64),
        torch.tensor([-5 / 3, 4 / 3, 0, 2 / 3], dtype=torch.float64),
    ],
    c_sol=torch.tensor([-1, 3 / 2, -1, 3 / 2, 0], dtype=torch.float64),
    c_error=torch.tensor([0, 1 / 2, -1, 1 / 2, 0], dtype=torch.float64),  # Placeholder
    )

DPS_C_MID = torch.tensor([-1, 1.25, -0.5, 1.25, 0], dtype=torch.float64)


class Bilal4Solver(RKAdaptiveStepsizeODESolver):
    order = 5
    tableau = _BILAL_4_TABLEAU
    mid = DPS_C_MID