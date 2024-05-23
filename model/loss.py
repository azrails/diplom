import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, eps=10e-6):
        super().__init__()
        self.register_buffer("margin", torch.tensor(margin))
        self.eps = eps

    def forward(self, x, y, labels):
        dist = nn.functional.pairwise_distance(x, y, keepdim=True)
        mdist = torch.clamp(self.margin - dist, min=0)
        loss = torch.mean(labels*torch.pow(dist + self.eps, 2) + (1 - labels)*torch.pow(mdist+self.eps, 2))
        return loss


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class ContrastiveSoftMax(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, image_embeddings, text_embeddings):
        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0
        return loss.mean()
