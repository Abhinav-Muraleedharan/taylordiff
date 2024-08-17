import math
import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn


def create_causal_mask(seq_length):
    """Create a causal mask for the attention mechanism."""
    return jnp.triu(jnp.ones((seq_length, seq_length)), k=1) == 0


# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        jnp.expand_dims(mask, axis=1)
    while mask.ndim < 4:
        jnp.expand_dims(mask, axis=0)
    return mask


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention



class MultiheadAttention(nn.Module):
    embed_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                )
        self.o_proj = nn.Dense(self.embed_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        # if mask is not None:
        #     mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o, attention

class PositionalEncoding(nn.Module):
    d_model : int         # Hidden dimensionality of the input.
    max_len : int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:,None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float

    @nn.compact
    def __call__(self, x, training):
        _, seq_length,_ = x.shape
        
        attention = MultiheadAttention(num_heads=self.n_heads,  embed_dim = self.d_model)
        attn_output,_ = attention(x,mask=create_causal_mask(seq_length))
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
        positional_encoding = PositionalEncoding(self.d_model)
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
        x = positional_encoding(x)
        
        for _ in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            )(x, training)
        
        logits = nn.Dense(self.vocab_size)(x)
        return logits