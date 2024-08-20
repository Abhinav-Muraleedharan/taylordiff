import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict

# Assuming the model is defined in a file named 'transformer_model.py'
from models.transformer import TransformerModel, TransformerBlock, MultiheadAttention, PositionalEncoding

@pytest.fixture
def model_config():
    return {
        'vocab_size': 1000,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 1024,
        'n_layers': 3,
        'dropout': 0.1
    }

def test_transformer_model_init(model_config):
    model = TransformerModel(**model_config)
    
    # Create a sample input
    batch_size, seq_length = 2, 10
    x = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    
    # Initialize the model
    key = jax.random.PRNGKey(0)
    params = model.init(key, x, training=True)

    assert 'params' in params

def test_transformer_model_output_shape(model_config):
    model = TransformerModel(**model_config)
    
    batch_size, seq_length = 2, 10
    x = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    
    key = jax.random.PRNGKey(0)
    params = model.init(key, x, training=True)
    
    output,_ = model.apply(params, x, training=False)
    
    assert output.shape == (batch_size, seq_length, model_config['vocab_size'])

def test_transformer_block():
    block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
    
    batch_size, seq_length, d_model = 2, 10, 256
    x = jnp.ones((batch_size, seq_length, d_model))
    
    key = jax.random.PRNGKey(0)
    params = block.init(key, x, training=True)
    
    output,_ = block.apply(params, x, training=False)
    
    assert output.shape == (batch_size, seq_length, d_model)

def test_multihead_attention():
    attention = MultiheadAttention(embed_dim=256, num_heads=8)
    
    batch_size, seq_length, embed_dim = 2, 10, 256
    x = jnp.ones((batch_size, seq_length, embed_dim))
    
    key = jax.random.PRNGKey(0)
    params = attention.init(key, x)
    
    output, _ = attention.apply(params, x)
    
    assert output.shape == (batch_size, seq_length, embed_dim)

def test_positional_encoding():
    pos_encoding = PositionalEncoding(d_model=256, max_len=100)
    
    batch_size, seq_length, d_model = 2, 10, 256
    x = jnp.ones((batch_size, seq_length, d_model))
    
    key = jax.random.PRNGKey(0)
    params = pos_encoding.init(key, x)
    
    output = pos_encoding.apply(params, x)
    
    assert output.shape == (batch_size, seq_length, d_model)
    assert not jnp.allclose(x, output)  # Ensure the encoding was added