from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ScaledDotProductAttention_MultiHead(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention_MultiHead, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            raise ValueError("Mask is not supported yet")

        # key, query, value shapes: [batch_size, num_heads, seq_len, dim]
        emb_dim = key.shape[-1]

        # Calculate attention weights
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            emb_dim
        )

        # masking
        if mask is not None:
            raise ValueError("Mask is not supported yet")

        # Softmax
        attention_weights = self.softmax(attention_weights)

        # modify value
        value = torch.matmul(attention_weights, value)

        return value, attention_weights


class PositionWiseFeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout: float = 0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # feed-forward network
        x = self.layer_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer_2(x)

        return x


class Add_and_Norm(nn.Module):

    def __init__(self, input_dim, dropout: Optional[float] = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x1, residual):
        x = x1
        # apply dropout of needed
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # add and then norm
        x = x + residual
        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, num_heads, dropout: Optional[float] = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        self.head_dim = input_dim // num_heads
        self.dropout = dropout

        # initialize weights
        self.query_w = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.keys_w = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.values_w = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.ff_layer_after_concat = nn.Linear(
            self.num_heads * self.head_dim, input_dim, bias=False
        )

        self.attention = ScaledDotProductAttention_MultiHead()

        if self.dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        # query, keys, values shapes: [batch_size, seq_len, input_dim]
        batch_size, len_query, len_keys, len_values = (
            queries.size(0),
            queries.size(1),
            keys.size(1),
            values.size(1),
        )

        # linear transformation before attention
        queries = (
            self.query_w(queries)
            .view(batch_size, len_query, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [batch_size, num_heads, seq_len, dim]
        keys = (
            self.keys_w(keys)
            .view(batch_size, len_keys, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [batch_size, num_heads, seq_len, dim]
        values = (
            self.values_w(values)
            .view(batch_size, len_values, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [batch_size, num_heads, seq_len, dim]

        # attention itself
        values, attention_weights = self.attention(
            queries, keys, values, mask=mask
        )  # values shape:[batch_size, num_heads, seq_len, dim]

        # concatenation
        out = (
            values.transpose(1, 2)
            .contiguous()
            .view(batch_size, len_values, self.num_heads * self.head_dim)
        )  # [batch_size, seq_len, num_heads * dim = input_dim]
        # go through last linear layer
        out = self.ff_layer_after_concat(out)

        return out


class EncoderLayer(nn.Module):

    def __init__(
        self,
        input_dim,
        num_heads,
        dropout: Optional[float] = 0.1,
        positional_encoding: bool = True,
    ):
        super(EncoderLayer, self).__init__()
        self.positional_encoding = positional_encoding
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.dropout = dropout

        # initialize layers
        self.self_attention = MultiHeadAttention(input_dim, num_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(
            input_dim, input_dim, dropout=dropout
        )
        self.add_norm_after_attention = Add_and_Norm(input_dim, dropout=dropout)
        self.add_norm_after_ff = Add_and_Norm(input_dim, dropout=dropout)

        # calculate positional encoding
        if self.positional_encoding:
            self.positional_encoding = PositionalEncoding(input_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # positional encoding
        if self.positional_encoding:
            x = self.positional_encoding(x)

        # multi-head attention
        residual = x
        x = self.self_attention(x, x, x)
        x = self.add_norm_after_attention(x, residual)

        # feed forward
        residual = x
        x = self.feed_forward(x)
        x = self.add_norm_after_ff(x, residual)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(
            1, 0, 2
        )  # [seq_len, batch_size, embedding_dim] -> [batch_size, seq_len, embedding_dim]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerLayer(nn.Module):

    def __init__(
        self,
        input_dim,
        num_heads,
        dropout: Optional[float] = 0.1,
        positional_encoding: bool = True,
    ):
        super(TransformerLayer, self).__init__()
        self.positional_encoding = positional_encoding
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.dropout = dropout

        # initialize layers
        self.self_attention = MultiHeadAttention(input_dim, num_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(
            input_dim, input_dim, dropout=dropout
        )
        self.add_norm_after_attention = Add_and_Norm(input_dim, dropout=dropout)
        self.add_norm_after_ff = Add_and_Norm(input_dim, dropout=dropout)

        # calculate positional encoding
        if self.positional_encoding:
            self.positional_encoding = PositionalEncoding(input_dim)

    def forward(self, key, value, query, mask=None):
        # key, value, and query shapes: [batch_size, seq_len, input_dim]
        # positional encoding
        if self.positional_encoding:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        # multi-head attention
        residual = query
        x = self.self_attention(queries=query, keys=key, values=value, mask=mask)
        x = self.add_norm_after_attention(x, residual)

        # feed forward
        residual = x
        x = self.feed_forward(x)
        x = self.add_norm_after_ff(x, residual)

        return x
