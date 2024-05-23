import math
import torch
from torch import nn
from .drop_path import DropPath

class ConvImgEmbedding(nn.Module):
    """
    Several convolutions and pooling of image before translate to embedding
    """
    def __init__(
            self,
            kernel_size: int = 7, stride: int = 1, padding: int = 0, dilation: int=1,
            pool_kernel_size: int=3, pool_stride: int=2, pool_padding: int=1,
            num_layers: int=1,
            input_channels_size: int=3,
            output_channels_size: int=64,
            hidden_channels_size: int=64,
            conv_bias: bool = False
    ):
        super().__init__()
        channels_list = [input_channels_size] + [hidden_channels_size for _ in range(num_layers - 1)] + [output_channels_size]
        self.conv_block = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels=channels_list[i], 
                              out_channels=channels_list[i + 1],
                              kernel_size=(kernel_size, kernel_size), 
                              stride=(stride, stride), 
                              padding=(padding, padding), 
                              bias=conv_bias, 
                              dilation=dilation),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)                    
                ) for i in range(num_layers)
            ]
        )
        self.flatening = nn.Flatten(2, 3)

    def get_result_size(self, channels_size, hight=224, width=224):
        res = self.forward(torch.zeros((1, channels_size, hight, width)))
        return res.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatening(self.conv_block(x)).transpose(-2, -1)


class TextEmbedding(nn.Module):
    def __init__(
            self,
            kernel_size: int = 3, stride: int = 1, padding: int = 0,
            pool_kernel_size: int=3, pool_stride: int=2, pool_padding: int=1,
            output_channels_size: int=384,
            hidden_channels_size: int=384,
            conv_bias: bool = False
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 
                              out_channels=output_channels_size,
                              kernel_size=(kernel_size, hidden_channels_size), 
                              stride=(stride, 1), 
                              padding=(padding, 0), 
                              bias=conv_bias
                              ),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size=(pool_kernel_size, 1), stride=(pool_stride, 1), padding=(pool_padding, 0))                    
        )
        
    def get_result_size(self, seq_len, input_dim):
        return self.forward(torch.zeros((1, seq_len, input_dim)))[0].shape[1]

    def forward_mask(self, mask):
        new_mask = mask.unsqueeze(1).float()
        cnn_weight = torch.ones(
            (1, 1, self.conv_layers[0].kernel_size[0]),
            device=mask.device,
            dtype=torch.float)
        new_mask = nn.functional.conv1d(
            new_mask, cnn_weight, None,
            self.conv_layers[0].stride[0], self.conv_layers[0].padding[0], 1, 1)
        new_mask = nn.functional.max_pool1d(
            new_mask, self.conv_layers[2].kernel_size[0],
            self.conv_layers[2].stride[0], self.conv_layers[2].padding[0], 1, False, False)
        new_mask = new_mask.squeeze(1)
        new_mask = (new_mask > 0)
        return new_mask

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        # print(f'fff {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        x = x.transpose(1, 3).squeeze(1)
        # print(f'ss {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        if mask is not None:
            mask = self.forward_mask(mask).unsqueeze(-1).float()
            x = x * mask
        return x, mask
    


class MultiHeadAttentionLayer(nn.Module):
    """
    Universal multihead attention layer
    Need attention needed key_len == value_len
    key == value == query - SelfAttention
    key == value, query - another - CrossAttention
    """
    def __init__(
            self, 
            input_size: int,  
            num_heads:int = 1, 
            attention_dropout: float=0.,
            projection_dropout: float=0.
            ):
        super().__init__()
        assert input_size % num_heads == 0
        self.head_size = input_size // num_heads
        self.input_size = input_size
        self.num_heads = num_heads


        self.q_mat = nn.Linear(self.input_size, self.input_size)
        self.k_mat = nn.Linear(self.input_size, self.input_size)
        self.v_mat = nn.Linear(self.input_size, self.input_size)

        self.multi_head_mat = nn.Linear(self.input_size, self.input_size)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)

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
        attention_probs = self.attention_dropout(attention_probs)

        #[batch_size, num_heads, query_len, key_len] -> [batch_size, num_heads, query_len, head_size]
        x = torch.matmul(attention_probs, V)

        #[batch_size, num_heads, query_len, head_size] -> [batch_size, query_len, num_heads, head_size]
        x = x.permute(0, 2, 1, 3).contiguous()
        #batch_size, query_len, num_heads, head_size] -> [batch_size, query_len, num_heads, head_size * num_heads]
        x = x.view(bs, -1, self.head_size * self.num_heads)
        #[batch_size, query_len, num_heads, head_size * num_heads] -> [batch_size, query_len, input_size]
        x = self.multi_head_mat(x)
        x = self.projection_dropout(x)
        return x, attention_probs


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with QKV self attention
    """
    def __init__(
            self, 
            embedding_dim: int, 
            feed_forward_dim: int=3000,
            num_heads: int=4, 
            attention_dropout: float=0.,
            projection_dropout: float=0.,
            feed_forward_dropout: float=0.1,
            drop_path_rate: float=0.1
        ):
        super().__init__()
        # self.attention = MultiHeadAttentionLayer(embedding_dim, num_heads, attention_dropout, projection_dropout)
        self.attention = Attention(embedding_dim, num_heads, attention_dropout, projection_dropout)
        # self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.mlp_layer = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(feed_forward_dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
            nn.Dropout(feed_forward_dropout)
        )
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.drop_path = DropPath(drop_path_rate)

    
    def forward(self, x: torch.Tensor, attention_mask: None|torch.Tensor = None):
        # print(f'in {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        x = x + self.drop_path(self.attention(self.norm_1(x)))
        # attended_x, attention_probs = self.attention(attended_x)
        # print(f'attn {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        attention_probs = None
        mlp_x = self.mlp_layer(self.norm_2(x))
        # print(f'mlp {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        x = x + self.drop_path(mlp_x)
        # print(f'skip {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        return x, attention_probs


class Encoder(nn.Module):
    """
    Encoder block, contains several transformer blocks
    """
    def __init__(
            self, 
            embedding_dim: int, 
            feed_forward_dim: int = 3000, 
            num_heads: int=4,
            blocks_num: int = 8,
            attention_dropout: float=0.,
            projection_dropout: float=0.,
            feed_forward_dropout: float=0.1,
        ):
        super().__init__()
        self.blocks = nn.ModuleList([
                TransformerBlock(
                    embedding_dim, 
                    feed_forward_dim, 
                    num_heads, 
                    attention_dropout, 
                    projection_dropout, 
                    feed_forward_dropout
                    )
                for _ in range(blocks_num)]
            )
        # self.apply(self.init_weight)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        attention_probs = []
        for block in self.blocks:
            x, attention_prob = block(x)
            attention_probs.append(attention_prob)
        return (x, attention_probs)

    # @staticmethod
    # def init_weight(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)



class PositionalEmbedding(nn.Module):
    """
    Add learnable start token and positional values to embeddings
    """
    def __init__(
            self,
            embedding_dim,
            number_of_embeddings
            ):
        super().__init__()
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dim), requires_grad=True
            )
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, number_of_embeddings + 1, embedding_dim), requires_grad=True
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((x, cls_token), dim=1)
        x = x + self.positional_embedding
        return x


class AttentionPooling(nn.Module):
    """
    Pooling vectors as weighted sum
    """
    def __init__(
           self,
           embedding_dim: int,
           attention_dropout: float=0.,
    ):
        super().__init__()
        self.attention_pool = nn.Linear(embedding_dim, 1)
        self.attention_dropout = nn.Dropout(attention_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #[batch_size, num_embeddings, embedding_dim] -> [batch_size, 1, num_embeddings]
        attention_probs = nn.functional.softmax(self.attention_pool(x), dim=1).transpose(-1, -2)
        attention_probs = self.attention_dropout(attention_probs)
        x = torch.matmul(attention_probs, x).squeeze(-2)
        return x


class StageOneEncoder(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        embedding_dim: int,
        embedding_dim_internal: int=320,
        mlp_ratio: int=2, 
        num_heads: int=4,
        num_blocks: int=8, 
        attention_dropout: float=0.,
        projection_dropout: float=0.,
        feed_forward_dropout: float=0.1,
        conv_kernel_size: int = 7,
        conv_num_layers: int = 2,
        conv_padding = None,
        conv_stride = None
    ):
        super().__init__()
        # for patching embedding
        conv_stride = conv_stride if conv_stride is not None else max(1, (conv_kernel_size // 2) - 1)
        conv_padding = conv_padding if conv_padding is not None else max(1, (conv_kernel_size // 2))
        feed_forward_dim = embedding_dim * mlp_ratio
        self.embedder = ConvImgEmbedding(
            padding=conv_padding,
            stride=conv_stride,
            kernel_size=conv_kernel_size,
            num_layers=conv_num_layers,
            output_channels_size=embedding_dim,
            hidden_channels_size=embedding_dim_internal,
        )
        number_of_embeddings = self.embedder.get_result_size(3, image_size[0], image_size[1])

        self.positional_encoder = PositionalEmbedding(embedding_dim, number_of_embeddings)
        self.encoder = Encoder(
            embedding_dim, 
            feed_forward_dim, 
            num_heads, 
            num_blocks, 
            attention_dropout, 
            projection_dropout, 
            feed_forward_dropout
            )

        self.attention_pooling = AttentionPooling(embedding_dim, attention_dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        # self.apply(self.init_weight)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_x = self.embedder(x)
        x = self.positional_encoder(conv_x)
        x, attention_probs = self.encoder(x)
        x = self.attention_pooling(x)
        x = self.fc(self.norm(x))
        return (x, attention_probs)

    # @staticmethod
    # def init_weight(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)


class ImgEncoder(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        embedding_dim: int,
        embedding_dim_internal: int=320,
        conv_kernel_size: int = 7,
        conv_num_layers: int = 2,
        conv_padding = None,
        conv_stride = None
    ):
        super().__init__()
        # for patching embedding
        conv_stride = conv_stride if conv_stride is not None else max(1, (conv_kernel_size // 2) - 1)
        conv_padding = conv_padding if conv_padding is not None else max(1, (conv_kernel_size // 2))
        self.embeder = ConvImgEmbedding(
            padding=conv_padding,
            stride=conv_stride,
            kernel_size=conv_kernel_size,
            num_layers=conv_num_layers,
            output_channels_size=embedding_dim,
            hidden_channels_size=embedding_dim_internal,
        )
        number_of_embeddings = self.embeder.get_result_size(3, image_size[0], image_size[1])
        self.positional_encoder = PositionalEmbedding(embedding_dim, number_of_embeddings)
        self.apply(self.init_weight)

    def forward(self, x):
        # print(f'img in {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        x = self.embeder(x)
        # print(f'img emb {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        x = self.positional_encoder(x)
        # print(f'img pos {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TextEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        embedding_dim_internal: int=320,
        mlp_ratio=3, 
        num_heads=4, 
        num_blocks= 6, 
        attention_dropout = .0, 
        projection_dropout = .0,
        feed_forward_dropout = 0.1

    ):
        super().__init__()
        # for patching embedding
        feed_forward_dim = mlp_ratio * embedding_dim_internal
        self.embeder = Encoder(
            embedding_dim_internal,
            feed_forward_dim, 
            num_heads, 
            num_blocks, 
            attention_dropout, 
            projection_dropout, 
            feed_forward_dropout
            )

        self.norm = nn.LayerNorm(embedding_dim_internal)
        self.fc = nn.Linear(embedding_dim_internal, embedding_dim)
        self.positional_encoder = PositionalEmbedding(embedding_dim, seq_len)
        self.apply(self.init_weight)

    def forward(self, x):
        x, _ = self.embeder(x)
        outp_x = self.fc(self.norm(x))
        return self.positional_encoder(outp_x)

    @staticmethod
    def length_to_mask(mask_lens):
        assert len(mask_lens.shape) == 1
        max_len = mask_lens.max().item()
        mask = torch.arange(max_len, device=mask_lens.device).expand(len(mask_lens), max_len) < mask_lens.unsqueeze(1)
        return mask.to(torch.uint8)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



class TextConvEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        embedding_dim_internal: int=320,
        conv_kernel_size: int = 7,
        conv_padding = None,
        conv_stride = None

    ):
        super().__init__()
        # for patching embedding
        conv_stride = conv_stride if conv_stride is not None else max(1, (conv_kernel_size // 2) - 1)
        conv_padding = conv_padding if conv_padding is not None else max(1, (conv_kernel_size // 2))
        self.embeder = TextEmbedding(
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            hidden_channels_size=embedding_dim_internal,
            output_channels_size=embedding_dim
        )
        number_of_embeddings = self.embeder.get_result_size(seq_len, embedding_dim_internal)
        self.positional_encoder = PositionalEmbedding(embedding_dim, number_of_embeddings)
        self.apply(self.init_weight)

    def forward(self, x):
        # print(f'fst text {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        x, _ = self.embeder(x)
        # print(f'second text {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        x = self.positional_encoder(x)
        # print(f'trd text {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        return x

    @staticmethod
    def length_to_mask(mask_lens):
        assert len(mask_lens.shape) == 1
        max_len = mask_lens.max().item()
        mask = torch.arange(max_len, device=mask_lens.device).expand(len(mask_lens), max_len) < mask_lens.unsqueeze(1)
        return mask.to(torch.uint8)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SiamEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_ratio: int=2, 
        num_heads: int=4,
        num_blocks: int=8, 
        attention_dropout: float=0.,
        projection_dropout: float=0.,
        feed_forward_dropout: float=0.1,
    ):
        super().__init__()
        feed_forward_dim = embedding_dim * mlp_ratio

        self.encoder = Encoder(
            embedding_dim, 
            feed_forward_dim, 
            num_heads, 
            num_blocks, 
            attention_dropout, 
            projection_dropout, 
            feed_forward_dropout
            )

        self.attention_pooling = AttentionPooling(embedding_dim, attention_dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.apply(self.init_weight)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'second {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        x, attention_probs = self.encoder(x)
        attn_x = self.attention_pooling(x)
        # print(f'attn_pool {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        outp_x = self.fc(self.norm(attn_x))
        # print(f'siam end {torch.isnan(x).any()}, {torch.isfinite(x).all()}')
        return (outp_x, attention_probs)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class LearnMetric(nn.Module):
    def __init__(self, embedding_dim=300, mlp_ratio=3):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, embedding_dim * mlp_ratio)
        self.activ = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim * mlp_ratio, 1)
        self.ouput_prob = nn.Sigmoid()
    
    def forward(self, x, y):
        z = torch.cat((x, y), dim=1)
        z = self.fc2(self.activ(self.fc(x)))
        return self.ouput_prob(z)
