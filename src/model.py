import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module): ...


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.Sequential(
            *[DecoderLayer(config) for _ in range(config.num_hidden_layers)],
        )
        self.norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, labels=None):
        _, seq_len = input_ids.shape

        input_embeds = self.embed_tokens(input_ids)
        # TODO Rotary position embedding
        position_embeds = self.embed_positions(
            torch.arange(seq_len, device=input_ids.device)
        )

        x = input_embeds + position_embeds
        x = self.dropout(x)
        x = self.layers(x)
        x = self.norm(x)

        logits = self.lm_head(x)
        loss = (
            None
            if labels is None
            else F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        )

        return logits, loss
