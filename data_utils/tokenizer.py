import torch
from transformers import AutoTokenizer

class Tokenizer:
    """
    Tokenizer wrapper, incapsulated parameters and transforms to uses device
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def tokenize(self, input_seq) -> tuple[torch.Tensor, torch.Tensor]:
        results = self.tokenizer(
            input_seq,
            truncation=True,
            return_tensors='pt', 
            padding=True
            )
        return (results['input_ids'], results['attention_mask'])

def get_auto_tokenizer(model_name: str) -> Tokenizer:
    """
    Download Bert tokenizer from hugFace for model and returns Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return Tokenizer(tokenizer)