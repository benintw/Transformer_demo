import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class CONFIG:
    d_model: int = 512
    input_dim: int = 512  # Set this to be equal to d_model
    num_heads: int = 8
    num_layers: int = 6
    batch_size: int = 30
    seq_len: int = 200
    dropout: float = 0.1


config = CONFIG()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model

        even_i = torch.arange(0, d_model, 2).float()
        denominator = torch.pow(10_000, even_i / d_model)
        position = torch.arange(max_seq_length).reshape(max_seq_length, 1).float()

        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)

        stacked = torch.stack([even_PE, odd_PE], dim=2).flatten(1, 2)
        self.register_buffer("PE", stacked.unsqueeze(0))

    def forward(self, x):
        return self.PE[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):

    def __init__(
        self, input_dim, d_model, context_length, dropout, num_heads, masking=True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.masking = masking

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(input_dim, d_model, bias=False)
        self.Wk = nn.Linear(input_dim, d_model, bias=False)
        self.Wv = nn.Linear(input_dim, d_model, bias=False)
        self.final_linear = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def split_heads(self, q, k, v, batch_size):
        q_len, k_len, v_len = q.size(1), k.size(1), v.size(1)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, k_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, v_len, self.num_heads, self.head_dim)
        return q, k, v

    def scaled_dot_product(self, q, k, v, num_tokens):
        attn_scores = q @ (k.transpose(2, 3))
        if self.masking:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / (self.d_model**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ v
        return context_vector

    def forward(self, inputs):
        print(f"MultiHeadAttention input shape: {inputs.shape}")
        batch_size, num_tokens, input_dim = inputs.shape

        queries = self.Wq(inputs)
        keys = self.Wk(inputs)
        values = self.Wv(inputs)

        queries, keys, values = self.split_heads(queries, keys, values, batch_size)

        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        context_vector = self.scaled_dot_product(queries, keys, values, num_tokens)

        context_vector = context_vector.permute(0, 2, 1, 3).contiguous()
        context_vector = context_vector.view(batch_size, num_tokens, self.d_model)

        output = self.final_linear(context_vector)
        print(f"MultiHeadAttention output shape: {output.shape}")
        return output


class MultiHeadCrossAttention(nn.Module):

    def __init__(
        self, input_dim, d_model, context_length, dropout, num_heads, masking=True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.masking = masking

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(input_dim, d_model, bias=False)
        self.Wk = nn.Linear(input_dim, d_model, bias=False)
        self.Wv = nn.Linear(input_dim, d_model, bias=False)
        self.final_linear = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def split_heads(self, q, k, v, batch_size):
        q_len, k_len, v_len = q.size(1), k.size(1), v.size(1)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, k_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, v_len, self.num_heads, self.head_dim)
        return q, k, v

    def scaled_dot_product(self, q, k, v, num_tokens):
        attn_scores = q @ (k.transpose(2, 3))
        if self.masking:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / (self.d_model**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ v
        return context_vector

    def forward(self, output_enc, output_decoder):
        print(
            f"MultiHeadCrossAttention input shapes: enc_output: {output_enc.shape}, dec_input: {output_decoder.shape}"
        )
        batch_size, dec_len, input_dim = output_decoder.shape
        _, enc_len, _ = output_enc.shape

        queries = self.Wq(output_decoder)
        keys = self.Wk(output_enc)
        values = self.Wv(output_enc)

        queries, keys, values = self.split_heads(queries, keys, values, batch_size)

        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        context_vector = self.scaled_dot_product(queries, keys, values, enc_len)

        context_vector = context_vector.permute(0, 2, 1, 3).contiguous()
        context_vector = context_vector.view(batch_size, dec_len, self.d_model)

        output = self.final_linear(context_vector)
        print(f"MultiHeadCrossAttention output shape: {output.shape}")
        return output


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        print(f"FeedForward input shape: {inputs.shape}")
        x = self.linear1(inputs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        print(f"FeedForward output shape: {x.shape}")
        return x


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(
            input_dim=cfg.d_model,
            d_model=cfg.d_model,
            context_length=cfg.seq_len,
            dropout=cfg.dropout,
            num_heads=cfg.num_heads,
            masking=False,
        )
        self.norm1 = LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.ff = FeedForward(cfg.d_model, cfg.dropout)
        self.norm2 = LayerNorm(cfg.d_model)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, inputs):
        print(f"EncoderBlock input shape: {inputs.shape}")
        shortcut = inputs
        x = self.mha(inputs)
        x = self.dropout1(x)
        x = shortcut + x
        x = self.norm1(x)

        shortcut2 = x
        x = self.ff(x)
        x = self.dropout2(x)
        x = shortcut2 + x
        output = self.norm2(x)
        print(f"EncoderBlock output shape: {output.shape}")
        return output


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.num_layers)])

    def forward(self, inputs):
        print(f"Encoder input shape: {inputs.shape}")
        for layer in self.layers:
            inputs = layer(inputs)
        print(f"Encoder output shape: {inputs.shape}")
        return inputs


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(
            input_dim=cfg.d_model,
            d_model=cfg.d_model,
            context_length=cfg.seq_len,
            dropout=cfg.dropout,
            num_heads=cfg.num_heads,
            masking=True,
        )
        self.norm1 = LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.cross_mha = MultiHeadCrossAttention(
            input_dim=cfg.d_model,
            d_model=cfg.d_model,
            context_length=cfg.seq_len,
            dropout=cfg.dropout,
            num_heads=cfg.num_heads,
            masking=False,
        )
        self.norm2 = LayerNorm(cfg.d_model)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.ff = FeedForward(cfg.d_model, cfg.dropout)
        self.norm3 = LayerNorm(cfg.d_model)
        self.dropout3 = nn.Dropout(cfg.dropout)

    def forward(self, enc_output, dec_input):
        print(
            f"DecoderBlock input shapes: enc_output: {enc_output.shape}, dec_input: {dec_input.shape}"
        )
        shortcut = dec_input
        dec_input = self.mha(dec_input)
        dec_input = self.dropout1(dec_input)
        dec_input = shortcut + dec_input
        dec_input = self.norm1(dec_input)

        shortcut2 = dec_input
        dec_input = self.cross_mha(enc_output, dec_input)
        dec_input = self.dropout2(dec_input)
        dec_input = shortcut2 + dec_input
        dec_input = self.norm2(dec_input)

        shortcut3 = dec_input
        dec_input = self.ff(dec_input)
        dec_input = self.dropout3(dec_input)
        dec_input = shortcut3 + dec_input
        output = self.norm3(dec_input)
        print(f"DecoderBlock output shape: {output.shape}")
        return output


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_layers)])

    def forward(self, enc_output, dec_input):
        print(
            f"Decoder input shapes: enc_output: {enc_output.shape}, dec_input: {dec_input.shape}"
        )
        for layer in self.layers:
            dec_input = layer(enc_output, dec_input)
        print(f"Decoder output shape: {dec_input.shape}")
        return dec_input


class Transformer(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.seq_len)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.output_linear = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * torch.sqrt(
            torch.tensor(self.cfg.d_model, dtype=torch.float32)
        )
        tgt = self.embedding(tgt) * torch.sqrt(
            torch.tensor(self.cfg.d_model, dtype=torch.float32)
        )

        src = src + self.pos_encoding(src)
        tgt = tgt + self.pos_encoding(tgt)

        print(f"Transformer input shapes: src: {src.shape}, tgt: {tgt.shape}")
        enc_output = self.encoder(src)
        dec_output = self.decoder(enc_output, tgt)
        output = self.output_linear(dec_output)
        print(f"Transformer output shape: {output.shape}")
        return output


def main():
    vocab_size = 50257
    transformer = Transformer(config, vocab_size)

    src = torch.randint(0, vocab_size, (config.batch_size, config.seq_len))
    tgt = torch.randint(0, vocab_size, (config.batch_size, config.seq_len // 2))

    output = transformer(src, tgt)
    print(f"Output shape: {output.shape}")

    last_token = output[:, -1, :]
    pred_token_id = torch.argmax(torch.softmax(last_token, dim=-1), dim=-1)
    print(pred_token_id)


if __name__ == "__main__":
    main()
