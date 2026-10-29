"""q=1 MobiZO training step."""
import torch
from torch import nn


class ZOTrainingStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, labels, state, noise, learning_rate, eps):
        # tokens/labels: [E, S]; actual transformer batch: [2E, S].
        logits = self.model(torch.cat((tokens, tokens)), state, noise, eps).float()
        batch, seq_len = tokens.shape
        logits = logits[:, :-1].reshape(2, batch, seq_len - 1, -1)
        targets = labels[:, 1:].to(torch.int64)
        valid = targets != -100
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
        indices = safe_targets.reshape(1, batch, seq_len - 1, 1).expand(2, -1, -1, -1)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = torch.gather(log_probs, -1, indices).squeeze(-1)
        losses = -(selected * valid).sum(dim=(1, 2)) / valid.sum().clamp(min=1)
        projected_grad = (losses[0:1] - losses[1:2]) / (2 * eps)
        updated_state = state - learning_rate * projected_grad * noise
        return losses, projected_grad, updated_state
