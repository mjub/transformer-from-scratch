import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module): ...


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.kv_proj = nn.Linear(
            config.hidden_size,
            2 * config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.scale = config.head_dim**-0.5
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    config.max_position_embeddings,
                    config.max_position_embeddings,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

    # (B, T, C) -> (B, T, C)
    def forward(self, x):
        B, T, C = x.shape
        G = self.config.num_attention_heads // self.config.num_key_value_heads

        q = (
            self.q_proj(x)
            .view(B, T, self.config.num_attention_heads, self.config.head_dim)
            .transpose(1, 2)
        )  # (B, H, T, D)

        # (B, T, 2 * Hk * D) -> 2 * (B, T, Hk * D)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        # (B, T, Hk * D) -> (B, T, Hk, D) -> (B, Hk, T, D) -> (B, G * Hk, T, D)
        k = (
            k.view(B, T, self.config.num_key_value_heads, self.config.head_dim)
            .transpose(1, 2)
            .repeat_interleave(G, dim=1)
        )

        # (B, T, Hk * D) -> (B, T, Hk, D) -> (B, Hk, T, D) -> (B, G * Hk, T, D)
        v = (
            v.view(B, T, self.config.num_key_value_heads, self.config.head_dim)
            .transpose(1, 2)
            .repeat_interleave(G, dim=1)
        )

        # (B, H, T, D) @ (B, G * Hk, D, T) -> (B, H, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, H, T, T) @ (B, G * Hk, T, D) -> (B, H, T, D)
        out = attn_weights @ v
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, H * D)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, self.config.num_attention_heads * self.config.head_dim)
        )

        # (B, T, H * D) -> (B, T, C)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        self.mlp = FeedForward(config)

    # (B, T, C) -> (B, T, C)
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    # (B, T) -> (B, T, V)
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape

        # (B, T) -> (B, T, C)
        input_embeds = self.embed_tokens(input_ids)
        # TODO Rotary position embedding
        # (T) -> (T, C)
        position_embeds = self.embed_positions(torch.arange(T, device=input_ids.device))

        # (B, T, C) + (T, C) -> (B, T, C)
        x = input_embeds + position_embeds
        x = self.dropout(x)
        for layer in self.layers:
            # TODO KV Caching
            x = layer(x)
        x = self.norm(x)

        # (B, T, C) @ (C, V) -> (B, T, V)
        logits = self.lm_head(x)
        loss = (
            None
            if labels is None
            else F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        )

        return logits, loss
