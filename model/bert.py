import torch
from torch import nn
from transformers import AutoModel

class TextEmbedding(nn.Module):
    """
    Bert model from hf with 2 fc layers if pool true
    If pool false returns tokens embeddings without processing from Bert
    """
    def __init__(self, model_name: str, pool:str = True, device:str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.pool = pool
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.normalize = torch.nn.functional.normalize
        for parameter in self.bert_model.parameters():
            parameter.requires_grad = False
    

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_seq: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        res = self.bert_model(input_seq, attention_mask=attention_mask)
        if self.pool is True:
            res = self.mean_pooling(res, attention_mask=attention_mask)
            return res
        return res
