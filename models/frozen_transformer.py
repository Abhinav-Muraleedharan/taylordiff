import math
import jax
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint
from flax import linen as nn
from .transformer import PositionalEncoding, TransformerBlock, TransformerModel




def create_model(config, vocab_size):
    model = TransformerModel(
        config['model']['type'],
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout']
    )
    return model



def load_model():
    # state = create_train_state(rng, config, vocab_size)
    checkpoint_path = '/home/abhi98m/backed_up/taylordiff/experiments/model_checkpoints/transformer/epoch_0'
    orbax_checkpointer =  orbax.checkpoint.PyTreeCheckpointer()
    weights =  orbax_checkpointer.restore(checkpoint_path)
    return weights["params"]




class FrozenMultiHeadAttention(nn.Module):
    attention_map: jnp.array
    d_model: int
    n_heads: int

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3*self.d_model,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                )
        self.o_proj = nn.Dense(self.d_model,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)



    def __call__(self, x):
        
        batch_size, seq_length, embed_dim = x.shape
        # if mask is not None:
        #     mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.n_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        values = jnp.einsum('bhtt,bhtd->bhtd', self.attention_map, v) # product of attention with value vectors
        values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        return o
    
    
class FrozenTransformerBlock(nn.Module):
    attention_map: jnp.array
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.attention = FrozenMultiHeadAttention(attention_map=self.attention_map, d_model=self.d_model, n_heads=self.n_heads)
        self.dropout_fun = nn.Dropout(rate=self.dropout)
        self.layer_norm_fun = nn.LayerNorm()
        self.dense_layer_1 = nn.Dense(self.d_ff)
        self.dense_layer_2 = nn.Dense(self.d_model)
        
    def __call__(self, x, training):
    
        attn_output = self.attention(x)
        x = x + self.dropout_fun(attn_output, deterministic=not training)
        x = self.layer_norm_fun(x)
        ff_output = self.dense_layer_1(x)
        ff_output = nn.gelu(ff_output)
        ff_output = self.dense_layer_2(ff_output)
        x = x + self.dropout_fun(ff_output, deterministic=not training)
        x = self.layer_norm_fun(x)

        return x


class FrozenTransformerModel(nn.Module):
    vocab_size: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float
    @nn.compact
    def __call__(self, x, training):
        params = load_model()
        base_model = create_model()
        _ ,attention_map_list = base_model.apply({'params': params}, x, training=True)
        positional_encoding = PositionalEncoding(self.d_model)
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
        x = positional_encoding(x)
        
        for i in range(self.n_layers):
            attention_map = attention_map_list[i]
            x = FrozenTransformerBlock(
                attention_map = attention_map,
                d_model=self.d_model,
                n_heads = self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            )(x, training)
        
        logits = nn.Dense(self.vocab_size)(x)
        return logits, attention_map_list
    
