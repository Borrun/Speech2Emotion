import torch
import torch.nn as nn

class WavLMScoreNet(nn.Module):
    """
    Input: wav [B, N] float32
    Output: logits [B, T] where T is encoder frames
    """
    def __init__(self, model_name="microsoft/wavlm-base-plus", freeze_backbone=True):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, 1)

    def forward(self, wav, attention_mask=None):
        # transformers expect [B, N]
        out = self.backbone(input_values=wav, attention_mask=attention_mask).last_hidden_state  # [B, T, H]
        logits = self.head(out).squeeze(-1)  # [B, T]
        return logits
