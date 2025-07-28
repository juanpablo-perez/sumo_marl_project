import torch
import torch.nn as nn

class Policy(nn.Module):
    """
    4-layer MLP (512 → 512) con LayerNorm, GELU y Dropout.
    Llaves en el checkpoint = 'body.*'  y  'head.*'
    """
    def __init__(self, obs: int, act: int, hid: int = 512, p: float = 0.1):
        super().__init__()
        layers = []
        in_f = obs
        for _ in range(4):
            layers += [
                nn.Linear(in_f, hid),
                nn.LayerNorm(hid),
                nn.GELU(),
                nn.Dropout(p),
            ]
            in_f = hid
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hid, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        return torch.softmax(self.head(z), dim=-1)
