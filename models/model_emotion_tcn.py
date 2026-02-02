import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dilation)

    def forward(self, x):
        # x: [B,C,T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualCausalBlock(nn.Module):
    def __init__(self, ch, k=3, dilation=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, k=k, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.net(x)


class EmotionTCN(nn.Module):
    """
    Input: mel [B, T, M]
    Output:
      type: [B, T, n_types]
      lvl : [B, T, n_levels]
      bnd : [B, T] or None
    """
    def __init__(
        self,
        n_mels=80,
        channels=128,
        layers=6,
        dropout=0.1,
        n_types=7,
        n_levels=6,
        use_boundary_head=True,
    ):
        super().__init__()
        self.n_types = int(n_types)
        self.n_levels = int(n_levels)
        self.use_boundary_head = bool(use_boundary_head)

        self.in_proj = nn.Conv1d(n_mels, channels, 1)

        blocks = []
        for i in range(int(layers)):
            d = 2 ** i
            blocks.append(ResidualCausalBlock(channels, k=3, dilation=d, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

        self.type_head = nn.Conv1d(channels, self.n_types, 1)
        self.lvl_head = nn.Conv1d(channels, self.n_levels, 1)
        self.bnd_head = nn.Conv1d(channels, 1, 1) if self.use_boundary_head else None

    def forward(self, mel):
        # mel: [B,T,M] -> [B,M,T]
        x = mel.transpose(1, 2)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)

        type_logits = self.type_head(x).transpose(1, 2)  # [B,T,n_types]
        lvl_logits = self.lvl_head(x).transpose(1, 2)    # [B,T,n_levels]

        if self.bnd_head is not None:
            bnd_logits = self.bnd_head(x).squeeze(1)     # [B,T]
        else:
            bnd_logits = None

        return {"type": type_logits, "lvl": lvl_logits, "bnd": bnd_logits}
