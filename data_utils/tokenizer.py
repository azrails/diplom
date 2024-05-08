import torch
from transformers import BertTokenizer

class Tokenizer:
    """
    Tokenizer wrapper, incapsulated parameters and transforms to uses device
    """
    def __init__(self, bert_tokenizer):
        self.tokenizer = bert_tokenizer
    
    def tokenize(self, input_seq) -> tuple[torch.Tensor, torch.Tensor]:
        results = self.tokenizer(
            input_seq, 
            truncation=True, 
            return_tensors='pt', 
            add_special_tokens=True,
            padding='max_length'
            )
        return (results['input_ids'], results['attention_mask'])

def get_bert_tokenizer(model_name: str) -> Tokenizer:
    """
    Download Bert tokenizer from hugFace for model and returns Tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return Tokenizer(tokenizer)