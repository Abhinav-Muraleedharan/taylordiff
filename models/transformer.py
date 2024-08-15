import jax.numpy as jnp
from flax import linen as nn

class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float

    @nn.compact
    def __call__(self, x, training):
        attention = nn.MultiHeadDotProductAttention(num_heads=self.n_heads)
        attn_output = attention(x, x, x)
        x = x + nn.Dropout(rate=self.dropout)(attn_output, deterministic=not training)
        x = nn.LayerNorm()(x)
        
        ff_output = nn.Dense(self.d_ff)(x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dense(self.d_model)(ff_output)
        x = x + nn.Dropout(rate=self.dropout)(ff_output, deterministic=not training)
        x = nn.LayerNorm()(x)
        
        return x

class TransformerModel(nn.Module):
    vocab_size: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float

    @nn.compact
    def __call__(self, x, training):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
        
        for _ in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            )(x, training)
        
        logits = nn.Dense(self.vocab_size)(x)
        return logits