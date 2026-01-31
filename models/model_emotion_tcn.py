import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class EmotionTCN(nn.Module):
    """
    Input : mel [B, T, M]
    Output:
      type_logits  [B, T, 6]
      lvl_logits   [B, T, 6]
      bnd_logits   [B, T]   (optional boundary head to stabilize step-wise outputs)
    """
    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 128,
        layers: int = 6,
        dropout: float = 0.1,
        n_types: int = 6,
        n_levels: int = 6,
        use_boundary_head: bool = True,
    ):
        super().__init__()
        self.use_boundary_head = bool(use_boundary_head)

        self.in_proj = nn.Conv1d(n_mels, channels, 1)

        blocks = []
        for i in range(layers):
            d = 2 ** i
            blocks.append(nn.Sequential(
                CausalConv1d(channels, channels, k=3, dilation=d),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, 1),
                nn.ReLU(),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.type_head = nn.Conv1d(channels, n_types, 1)
        self.lvl_head = nn.Conv1d(channels, n_levels, 1)
        self.bnd_head = nn.Conv1d(channels, 1, 1) if self.use_boundary_head else None

    def forward(self, mel: torch.Tensor) -> dict:
        x = mel.transpose(1, 2)  # [B, M, T]
        x = self.in_proj(x)
        for blk in self.blocks:
            x = x + blk(x)

        type_logits = self.type_head(x).transpose(1, 2)  # [B, T, 6]
        lvl_logits = self.lvl_head(x).transpose(1, 2)    # [B, T, 6]

        if self.use_boundary_head:
            bnd_logits = self.bnd_head(x).squeeze(1)     # [B, T]
        else:
            bnd_logits = None

        return {"type": type_logits, "lvl": lvl_logits, "bnd": bnd_logits}
