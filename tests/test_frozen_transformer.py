import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn

# Import your modules
from models.frozen_transformer import FrozenMultiHeadAttention, FrozenTransformerBlock, FrozenTransformerModel, create_model, load_model

@pytest.fixture
def config():
    return {
        'model': {
            'type': 'transformer',
            'd_model': 256,
            'n_heads': 8,
            'd_ff': 1024,
            'n_layers': 3,
            'dropout': 0.1
        }
    }

@pytest.fixture
def vocab_size():
    return 1000

@pytest.fixture
def sample_input(vocab_size):
    batch_size, seq_length = 2, 10
    return jnp.ones((batch_size, seq_length), dtype=jnp.int32)

def test_frozen_multihead_attention(config):
    batch_size, seq_length = 2, 10
    d_model = config['model']['d_model']
    n_heads = config['model']['n_heads']
    
    attention_map = jnp.ones((batch_size, n_heads, seq_length, seq_length))
    
    fmha = FrozenMultiHeadAttention(attention_map=attention_map, d_model=d_model, n_heads=n_heads)
    
    x = jnp.ones((batch_size, seq_length, d_model))
    
    key = jax.random.PRNGKey(0)
    params = fmha.init(key, x)
    
    output = fmha.apply(params, x)
    
    assert output.shape == (batch_size, seq_length, d_model)
    print(f"FrozenMultiHeadAttention output shape: {output.shape}")

def test_frozen_transformer_block(config):
    batch_size, seq_length = 2, 10
    d_model = config['model']['d_model']
    n_heads = config['model']['n_heads']
    d_ff = config['model']['d_ff']
    dropout = config['model']['dropout']
    
    attention_map = jnp.ones((batch_size, n_heads, seq_length, seq_length))
    
    ftb = FrozenTransformerBlock(attention_map=attention_map, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
    
    x = jnp.ones((batch_size, seq_length, d_model))
    
    key = jax.random.PRNGKey(0)
    params = ftb.init(key, x, training=False)
    
    output = ftb.apply(params, x, training=False)
    
    assert output.shape == (batch_size, seq_length, d_model)
    print(f"FrozenTransformerBlock output shape: {output.shape}")

def test_frozen_transformer_model(config, vocab_size, sample_input, monkeypatch):
    # Mock the load_model function
    def mock_load_model():
        return {'params': jnp.ones((1,))}
    monkeypatch.setattr('models.frozen_transformer.load_model', mock_load_model)
    
    # Mock the create_model function
    def mock_create_model():
        return nn.Module()
    monkeypatch.setattr('models.frozen_transformer.create_model', mock_create_model)
    
    # Mock the base_model.apply function
    def mock_apply(*args, **kwargs):
        batch_size, seq_length = sample_input.shape
        n_heads = config['model']['n_heads']
        n_layers = config['model']['n_layers']
        attention_maps = [jnp.ones((batch_size, n_heads, seq_length, seq_length)) for _ in range(n_layers)]
        return None, attention_maps
    monkeypatch.setattr(nn.Module, 'apply', mock_apply)
    
    ftm = FrozenTransformerModel(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout']
    )
    
    key = jax.random.PRNGKey(0)
    params = ftm.init(key, sample_input, training=False)
    
    output,attention_map_list = ftm.apply(params, sample_input, training=False)
    print(output)
    print("Attention Map list:",attention_map_list)
    
    assert output.shape == (sample_input.shape[0], sample_input.shape[1], vocab_size)
    print(f"FrozenTransformerModel output shape: {output.shape}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])