
import torch

from dataclasses import dataclass

from .penc import PositionalEncoding



@dataclass
class AttentionClassifierConfig:
    vocab_size: int
    d_model: int
    dropout: float = 0.0
    p_masking: float = 0.0
    mask_token_id: int = 4


@dataclass
class ClassifierConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float = 0.0
    p_masking: float = 0.0
    mask_token_id: int = 4


@dataclass
class LSTMClassifierConfig:
    vocab_size: int
    d_embedding: int
    d_lstm: int
    n_layers: int = 2
    dropout: float = 0.0


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.w_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.score_dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        s = torch.bmm(q, k.transpose(-2, -1))
        s = self.score_dropout(s)

        if mask is not None:
            mask = -1e9 * mask.float()
            s += mask.unsqueeze(1)
        s = torch.softmax(s, dim=-1)

        return s @ v, s


class AttentionClassifier(torch.nn.Module):
    def __init__(self, config: AttentionClassifierConfig):
        super().__init__()

        self.config = config

        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)

        self.sa = SelfAttention(config.d_model)

        self.ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(config.d_model),
            torch.nn.Linear(config.d_model, config.d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout))

        self.head = torch.nn.Linear(config.d_model, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.config.p_masking > 0:
            idx = torch.rand(x.shape, device=x.device) < self.config.p_masking
            x[idx & (x > self.config.mask_token_id)] = self.config.mask_token_id

        x = self.embedding(x)
        x, s = self.sa(x, mask=mask)
        x = self.ffn(x[:, 0, :])
        x = self.head(x)
        return x, [s]


class EncoderBlock(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.0):
        super().__init__()

        self.ln0 = torch.nn.LayerNorm(d_model)
        self.mha = torch.nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)

        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        xn = self.ln0(x)
        x_attn, scores = self.mha(xn, xn, xn, key_padding_mask=mask, need_weights=True, average_attn_weights=True)
        x = x + x_attn

        x = x + self.ffn(self.ln1(x))
        return x, scores


class TransformerClassifier(torch.nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()

        self.config = config

        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.pre_norm = torch.nn.LayerNorm(config.d_model)
        self.positional_encoding = PositionalEncoding(d_model=config.d_model, max_sequence_length=1024)

        self.blocks = torch.nn.ModuleList([
            EncoderBlock(d_model=config.d_model, n_heads=config.n_heads, dropout=config.dropout) for _ in range(config.n_layers)
        ])

        self.post_norm = torch.nn.LayerNorm(config.d_model)
        self.head = torch.nn.Linear(config.d_model, 2)

        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.config.p_masking > 0:
            idx = torch.rand(x.shape, device=x.device) < self.config.p_masking
            x[idx & (x > self.config.mask_token_id)] = self.config.mask_token_id

        x = self.embedding(x)
        x = self.pre_norm(x)
        x = self.positional_encoding(x)

        scores = []
        for block in self.blocks:
            x, s = block(x, mask=mask)
            scores.append(s)

        x = x[:, 0, :]
        x = self.post_norm(x)
        x = self.head(x)

        return x, scores


class LSTMClassifier(torch.nn.Module):
    def __init__(self, config: LSTMClassifierConfig):
        super().__init__()

        self.config = config

        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_embedding)
        self.lstm = torch.nn.LSTM(
            input_size=config.d_embedding,
            hidden_size=config.d_lstm,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True)
        
        self.head = torch.nn.Linear(2 * config.d_lstm, 2)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.lstm(x)
        #x = x[:, 0, :]
        x1 = torch.mean(x, dim=1)
        x2, _ = torch.max(x, dim=1)
        x = x1 + x2
        x = self.head(x)
        return x, None


