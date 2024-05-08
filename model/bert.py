import torch
from torch import nn
from transformers import BertModel

class BertEmbedding(nn.Module):
    """
    Bert model from hf with 2 fc layers if pool true
    If pool false returns tokens embeddings without processing from Bert
    """
    def __init__(self, model_name: str, pool:str = True, device:str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.pool = pool
        self.bert_model = BertModel.from_pretrained(model_name)
        for parameter in self.bert_model.parameters():
            parameter.requires_grad = False
        # self.fc1 = nn.Linear(768, 768, bias=True)
        # self.activation_fn = torch.nn.GELU()
        # self.fc2 = nn.Linear(768, 768)


    def forward(self, input_seq: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        res = self.bert_model(input_seq, attention_mask=attention_mask)
        if self.pool:
            # x = res.pooler_output
            # x = self.fc1(x)
            # x = self.fc2(self.activation_fn(x))
            return res.pooler_output
        return res.last_hidden_state
