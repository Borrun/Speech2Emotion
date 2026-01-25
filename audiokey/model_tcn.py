import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dilation)

    def forward(self, x):
        # x [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class TCNScoreNet(nn.Module):
    """
    Input: mel [B, T, M]
    Output: logits [B, T]  (higher => more likely key)
    """
    def __init__(self, n_mels=80, channels=128, layers=6, dropout=0.1):
        super().__init__()
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
        self.out = nn.Conv1d(channels, 1, 1)

    def forward(self, mel):
        x = mel.transpose(1, 2)  # [B, M, T]
        x = self.in_proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        logits = self.out(x).squeeze(1)  # [B, T]
        return logits
