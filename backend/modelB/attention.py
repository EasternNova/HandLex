import torch
from torch import nn
import torch.nn.functional as F


class BaseAttention(nn.Module):
    """
    Shared implementation for ModelB attention layers.

    Input/output representation:
        (batch, sequence_length, d_model)

    Internally:
        (batch, heads, sequence_length, head_dim)
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads "
                f"(got d_model={d_model}, "
                f"num_heads={num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

    def _split_heads(
        self,
        states,
        batch_size,
        sequence_length,
    ):
        """
        Convert:

            (B, T, D)

        into:

            (B, H, T, Dh)
        """

        return (
            states
            .view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    def _merge_heads(
        self,
        states,
        batch_size,
        sequence_length,
    ):
        """
        Convert:

            (B, H, T, Dh)

        back into:

            (B, T, D)
        """

        return (
            states
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.d_model,
            )
        )

    def _apply_attention(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
    ):
        """
        Compute scaled dot-product attention.
        """

        attn_weights = torch.matmul(
            query_states,
            key_states.transpose(-1, -2),
        )

        if attention_mask is not None:
            attn_weights = (
                attn_weights + attention_mask
            )

        attn_weights = F.softmax(
            attn_weights,
            dim=-1,
        )

        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )

        return torch.matmul(
            attn_probs,
            value_states,
        )


class SelfAttention(BaseAttention):

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        """
        Self-attention.

        Input:
            hidden_states: (B, T, D)

        Output:
            (B, T, D)
        """

        batch_size, tgt_len, _ = (
            hidden_states.size()
        )

        query_states = (
            self.q_proj(hidden_states)
            * self.scaling
        )

        key_states = self.k_proj(
            hidden_states
        )

        value_states = self.v_proj(
            hidden_states
        )

        query_states = self._split_heads(
            query_states,
            batch_size,
            tgt_len,
        )

        key_states = self._split_heads(
            key_states,
            batch_size,
            tgt_len,
        )

        value_states = self._split_heads(
            value_states,
            batch_size,
            tgt_len,
        )

        attn_output = self._apply_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
        )

        attn_output = self._merge_heads(
            attn_output,
            batch_size,
            tgt_len,
        )

        return self.out_proj(attn_output)


class CrossAttention(BaseAttention):

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
    ):
        """
        Cross-attention.

        Query:
            decoder hidden states

        Key/value:
            encoder hidden states

        Input:
            hidden_states:    (B, Tq, D)
            key_value_states: (B, Tk, D)

        Output:
            (B, Tq, D)
        """

        batch_size, tgt_len, _ = (
            hidden_states.size()
        )

        src_len = key_value_states.size(1)

        query_states = (
            self.q_proj(hidden_states)
            * self.scaling
        )

        key_states = self.k_proj(
            key_value_states
        )

        value_states = self.v_proj(
            key_value_states
        )

        query_states = self._split_heads(
            query_states,
            batch_size,
            tgt_len,
        )

        key_states = self._split_heads(
            key_states,
            batch_size,
            src_len,
        )

        value_states = self._split_heads(
            value_states,
            batch_size,
            src_len,
        )

        attn_output = self._apply_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
        )

        attn_output = self._merge_heads(
            attn_output,
            batch_size,
            tgt_len,
        )

        return self.out_proj(attn_output)


class CausalSelfAttention(BaseAttention):

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        """
        Causal self-attention.

        Each position can attend only to
        itself and previous positions.

        Input:
            (B, T, D)

        Output:
            (B, T, D)
        """

        batch_size, tgt_len, _ = (
            hidden_states.size()
        )

        query_states = (
            self.q_proj(hidden_states)
            * self.scaling
        )

        key_states = self.k_proj(
            hidden_states
        )

        value_states = self.v_proj(
            hidden_states
        )

        query_states = self._split_heads(
            query_states,
            batch_size,
            tgt_len,
        )

        key_states = self._split_heads(
            key_states,
            batch_size,
            tgt_len,
        )

        value_states = self._split_heads(
            value_states,
            batch_size,
            tgt_len,
        )

        # -------------------------------------------------
        # Causal mask
        # -------------------------------------------------

        causal_mask = torch.triu(
            torch.ones(
                tgt_len,
                tgt_len,
                device=hidden_states.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        attn_weights = torch.matmul(
            query_states,
            key_states.transpose(-1, -2),
        )

        attn_weights = attn_weights.masked_fill(
            causal_mask.view(
                1,
                1,
                tgt_len,
                tgt_len,
            ),
            float("-inf"),
        )

        # -------------------------------------------------
        # Additional padding mask
        # -------------------------------------------------

        if attention_mask is not None:
            attn_weights = (
                attn_weights + attention_mask
            )

        attn_weights = F.softmax(
            attn_weights,
            dim=-1,
        )

        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )

        attn_output = torch.matmul(
            attn_probs,
            value_states,
        )

        attn_output = self._merge_heads(
            attn_output,
            batch_size,
            tgt_len,
        )

        return self.out_proj(attn_output)