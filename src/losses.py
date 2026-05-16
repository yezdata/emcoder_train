import torch
import torch.nn as nn
import torch.nn.functional as F

from src.types_shapes import ClassLabels, ClassLogits



class AsymmetricLoss(nn.Module):
    def __init__(
        self, 
        gamma_neg: float = 4.0, 
        gamma_pos: float = 0.0, 
        clip: float = 0.05, 
        eps: float = 1e-8
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps


    def forward(self, logits: ClassLogits, labels: ClassLabels, reduction: str | None = 'mean') -> torch.Tensor:
        labels = labels.to(logits.dtype)
        
        probs = torch.sigmoid(logits)
        
        log_pos = F.logsigmoid(logits)
        pos_weight = torch.exp(F.logsigmoid(-logits) * self.gamma_pos)
        loss_pos = -labels * pos_weight * log_pos
        
        pm = torch.clamp(probs - self.clip, min=0.0)
        
        neg_weight = (pm + self.eps).pow(self.gamma_neg)
        
        log_neg_shifted = torch.log(1.0 - pm + self.eps)
        
        loss_neg = -(1.0 - labels) * neg_weight * log_neg_shifted
        
        loss = loss_pos + loss_neg
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss


    def compute_bayesian_loss(self, logits_samples: torch.Tensor, labels: ClassLabels) -> torch.Tensor:
        n_samples, batch_size, num_classes = logits_samples.shape
        labels_expanded = labels.unsqueeze(0).expand(n_samples, batch_size, num_classes)
        
        loss_samples = self.forward(logits_samples, labels_expanded, reduction=None)
        
        return loss_samples.mean(dim=0).mean()




# weighted BCE with logits loss
    

    # criterion = WeightedBCEWithLogitsLoss(
    #     device=accelerator.device, pos_weights=weights
    # )
    # # computing the weights of labels
    # pos_counts = np.sum(np.array(tokenized_train_ds["labels"]), axis=0)
    # neg_counts = len(tokenized_train_ds) - pos_counts

    # # NOTE: USE SQRT / LOG FOR NUM STABILITY
    # # - 1e-5 avoids division by zero
    # if train_config.loss_weights == "sqrt":
    #     raw_weights = np.sqrt((neg_counts + 1e-5) / (pos_counts + 1e-5))
    # elif train_config.loss_weights == "log":
    #     raw_weights = 1 + np.log((neg_counts + 1e-5) / (pos_counts + 1e-5))
    # else:
    #     raw_weights = (neg_counts + 1e-5) / (pos_counts + 1e-5)

    # # PREVENT EXTREME WEIGTHS
    # raw_weights = np.clip(raw_weights, a_min=0.1, a_max=20.0)
    # weights = torch.tensor(raw_weights, dtype=torch.float32)

    # print("Label Weights:", weights)


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, device, pos_weights=None):
        super().__init__()
        self.pos_weights = pos_weights.to(device) if pos_weights is not None else None

        self.loss_fct_none = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights, reduction='none')
        self.loss_fct_mean = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights, reduction='mean')

    def forward(self, logits: ClassLogits, labels: ClassLabels) -> torch.Tensor:
        return self.loss_fct_mean(logits, labels.to(logits.dtype))

    def compute_bayesian_loss(
        self, logits_samples: torch.Tensor, labels: ClassLabels
    ) -> torch.Tensor:
        labels = labels.to(logits_samples.dtype)

        n_samples = logits_samples.size(0)

        labels_expanded = labels.unsqueeze(0).expand(n_samples, -1, -1)

        loss_samples = self.loss_fct_none(logits_samples, labels_expanded) # (n_samples, B, num_labels)

        loss = loss_samples.mean(dim=0).mean()

        return loss



