import torch
import torch.nn as nn


class SASRec(nn.Module):
    """
    Minimal SASRec-style model for next-item prediction.

    Input:
        input_seq: tensor of shape [batch_size, max_len]

    Output:
        logits: tensor of shape [batch_size, num_items + 1]
    """

    def __init__(
        self,
        num_items,
        max_len=5,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim

        # +1 because we use 0 as padding
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=hidden_dim,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=hidden_dim,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, input_seq):
        """
        input_seq: [B, L]
        returns logits: [B, num_items + 1]
        """
        device = input_seq.device
        batch_size, seq_len = input_seq.shape

        # Position ids: [0, 1, 2, ..., L-1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        # Embeddings
        item_emb = self.item_embedding(input_seq)              # [B, L, H]
        pos_emb = self.position_embedding(positions)           # [B, L, H]
        x = item_emb + pos_emb
        x = self.dropout(x)

        # Padding mask: True where padding exists
        padding_mask = (input_seq == 0)                        # [B, L]

        # Causal mask so positions cannot look ahead
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()                                               # [L, L]

        # Transformer
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        x = self.layer_norm(x)

        # Get the hidden state at the last non-padding position
        lengths = (input_seq != 0).sum(dim=1).clamp(min=1)     # [B]
        last_indices = lengths - 1                             # [B]

        last_hidden = x[torch.arange(batch_size, device=device), last_indices]  # [B, H]

        # Score against all items
        logits = last_hidden @ self.item_embedding.weight.T    # [B, num_items + 1]

        # Never predict padding token
        logits[:, 0] = -1e9

        return logits