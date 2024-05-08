import math
import torch
from torch import nn

class PatchImageEmbedding(nn.Module):
    """
    Implimintation forward convolution image to embeddings
    hidden_size - size of embedding vector
    """
    def __init__(self, image_size: tuple[int, int], segment_size: int, hidden_size: int, channels_size: int):
        super().__init__()
        self.num_embeddings = ((image_size[0] // segment_size) + (image_size[1] // segment_size))
        self.embeder = nn.Conv2d(channels_size, hidden_size, stride=segment_size, kernel_size=segment_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #[batch_size, channels_size, image_hight, image_width] -> [batch_size, hidden_size, image_hight//segment_size, image_width//segment_size]
        x = self.embeder(x)
        #[batch_size, hidden_size, image_hight//segment_size, image_width//segment_size] -> [batch_size, num_embeddings, hidden_size]
        x = x.flatten(2).transpose(2, 1)
        return x


class ImageEmbedding(nn.Module):
    """
    Wraped ImageEmbedding and added positional embeding to start of each vector
    """
    def __init__(self, image_size: tuple[int, int], segment_size: int, hidden_size: int, channels_size: int, dropout:float=0.2):
        super().__init__()
        self.dropout_prob = dropout
        self.image_embedding = PatchImageEmbedding(image_size, segment_size, hidden_size, channels_size)
        self.sos = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, self.image_embedding.num_embeddings + 1, hidden_size))
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #[batch_size, channels_size, image_hight, image_width] -> [batch_size, num_of_embeddings + 1, hidden_size]
        x = self.image_embedding(x)
        bs = x.size[0]
        sos = self.sos.expand(bs, -1, -1)
        # cat start of sequence token to embeddings
        x = torch.cat((sos, x), dim=1)
        x = x + self.positional_embedding
        x = self.dropout(x)
        return x


class MultiHeadAttentionLayer(nn.Module):
    """
    Universal multihead attention layer
    Need attention needed key_len == value_len
    key == value == query - SelfAttention
    key == value - CrossAttention
    """
    def __init__(
            self, 
            input_size: int,  
            num_heads:int = 1, 
            head_size: int = None, 
            dropout:float = 0.2, 
            bias:bool=True,
            ):
        super().__init__()
        if head_size is None:
            self.head_size = input_size // num_heads
        self.input_size = input_size
        self.num_heads = num_heads


        self.q_mat = nn.Linear(self.input_size, self.num_heads * self.head_size, bias=bias)
        self.k_mat = nn.Linear(self.input_size, self.num_heads * self.head_size, bias=bias)
        self.v_mat = nn.Linear(self.input_size, self.num_heads * self.head_size, bias=bias)

        self.multi_head_mat = nn.Linear(self.num_heads * self.input_size, self.input_size)
    
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            attention_mask:None|torch.Tensor = None, 
            ) -> torch.Tensor:
        
        bs = query.size()[0]
        #[batch_size, query_len, input_size] -> [batch_size, query_len, head_size * num_heads]
        Q = self.q_mat(query)
        #[batch_size, key_len, input_size] -> [batch_size, key_len, head_size * num_heads]
        K = self.k_mat(key)
        #[batch_size, value_len, input_size] -> [batch_size, value_len, head_size * num_heads]
        V = self.v_mat(value)

        #[batch_size, query_len, head_size * num_heads] -> [batch_size, num_heads, query_len, head_size]
        #Про эт можно думать, что это были склеенные конкатенацией головы, которые мы потом разорвали
        #и перевернули так, что бы голова соответсвовала входным значениям т.е грубо говоря просто список голов
        Q = Q.view(bs, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(bs, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(bs, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        #[batch_size, num_heads, query_len, head_size] -> [batch_size, num_heads, query_len, key_len]
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.num_heads * self.head_size)

        #masked for uses for example in decoder
        #Т.е пусть у нас для предложения x токенов, а максимально в батче y
        #тогда y - x токенов будут заполнителями, а мы не хотим что бы они как - то участвовали в процессе обучения
        #при помощи маски заполним их маленькими числами
        if attention_mask is not None:
            attention_scores.masked_fill_(attention_mask[:, None, None, :] == 0, 1e-10)

        #dim = -1 т.к. извлекаем наиболее необходимую информацию для конкретного вектора из каждой головы
        #Т.к attention_scores - по сути матрица весов, то мы как бы подгоняем насколько конкретный 
        # вес -> вектор важен для понимания текущего вектора
        attention_probs = nn.functional.softmax(attention_scores, dim = -1)

        #[batch_size, num_heads, query_len, key_len] -> [batch_size, num_heads, query_len, head_size]
        x = torch.matmul(self.dropout(attention_probs), V)

        #[batch_size, num_heads, query_len, head_size] -> [batch_size, query_len, num_heads, head_size]
        x = x.permute(0, 2, 1, 3).contiguous()
        #batch_size, query_len, num_heads, head_size] -> [batch_size, query_len, num_heads, head_size * num_heads]
        x = x.view(bs, -1, self.head_size * self.num_heads)
        #[batch_size, query_len, num_heads, head_size * num_heads] -> [batch_size, query_len, input_size]
        x = self.multi_head_mat(x)
        return x, attention_probs


class MLPLayer(nn.Module):
    """
    2 layer perceptron for transformer block
    """
    def __init__(
            self, 
            input_size: int, 
            hidden_size: int, 
            dropout: float=0.2, 
        ):

        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(
            self, 
            input_size: int, 
            hidden_size: int, 
            num_heads: int=1, 
            dropout: float=0.2, 
            bias: bool=True,
        ):
        super().__init__()
        self.attention = MultiHeadAttentionLayer(input_size, num_heads, dropout=dropout, bias=bias)
        self.mlp = MLPLayer(input_size, hidden_size, dropout)
        self.norm_1 = nn.LayerNorm(input_size)
        self.norm_2 = nn.LayerNorm(input_size)

    
    def forward(self, x: torch.Tensor, attention_mask: None|torch.Tensor = None):
        attended_x = self.norm_1(x)
        attended_x, attention_probs = self.attention(attended_x, attended_x, attended_x, attention_mask)
        x = x + attended_x
        mlp_x = self.mlp(self.norm_2(x))
        x = x + mlp_x
        return x, attention_probs


class Encoder(nn.Module):
    def __init__(
            self, 
            input_size: int, 
            hidden_size: int, 
            blocks_num: int,
            num_heads: int=1, 
            dropout: float=0.2, 
            bias: bool=True,
        ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(blocks_num):
            self.blocks.append(
                Transformer(input_size, hidden_size, num_heads, dropout, bias)
            )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        attention_probs = []
        for block in self.blocks:
            x, attention_prob = block(x)
            attention_probs.append(attention_prob)
        return (x, attention_probs)


class StageOneEncoder(nn.Module):
    def __init__(
            self,
            image_size: tuple[int, int],
            segment_size: int,
            embedding_size: int, 
            hidden_size: int, 
            blocks_num: int,
            num_heads: int=1, 
            dropout: float=0.2, 
            bias: bool=True
        ):
        super().__init__()
        self.embedder = ImageEmbedding(image_size, segment_size, embedding_size, 3, dropout)
        self.encoder = Encoder(embedding_size, hidden_size, blocks_num, num_heads, dropout, bias)
        self.image_embedding = nn.Linear(embedding_size, 768, bias)
    
    def forward(self, x):
        x = self.embedder(x)
        x, attention_probs = self.encoder(x)
        x = self.image_embedding(x[:, 0])
        return (x, attention_probs)
