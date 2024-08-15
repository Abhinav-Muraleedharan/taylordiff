import jax.numpy as jnp
from flax import linen as nn

class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float

    @nn.compact
    def __call__(self, x, training, return_attention=False):
        attn_output, attention_weights = nn.MultiHeadAttention(num_heads=self.n_heads)(x, x, x, return_attention=True)
        x = x + nn.Dropout(rate=self.dropout)(attn_output, deterministic=not training)
        x = nn.LayerNorm()(x)
        
        ff_output = nn.Dense(self.d_ff)(x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dense(self.d_model)(ff_output)
        x = x + nn.Dropout(rate=self.dropout)(ff_output, deterministic=not training)
        x = nn.LayerNorm()(x)
        
        if return_attention:
            return x, attention_weights
        return x

class TransformerModel(nn.Module):
    vocab_size: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float

    @nn.compact
    def __call__(self, x, training, return_attention=False):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
        
        attention_weights = []
        for _ in range(self.n_layers):
            if return_attention:
                x, attn = TransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout
                )(x, training, return_attention=True)
                attention_weights.append(attn)
            else:
                x = TransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout
                )(x, training)
        
        logits = nn.Dense(self.vocab_size)(x)
        
        if return_attention:
            return logits, attention_weights
        return logits