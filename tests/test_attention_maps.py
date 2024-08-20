import pytest
import jax
import jax.numpy as jnp
import orbax.checkpoint
import matplotlib.pyplot as plt
import seaborn as sns
from flax import linen as nn

# Assuming the model is defined in a file named 'transformer_model.py'
from models.transformer import TransformerModel, TransformerBlock, MultiheadAttention, PositionalEncoding

@pytest.fixture
def model_config():
    return {
        'vocab_size': 1000,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 1024,
        'n_layers': 2,
        'dropout': 0.1
    }


def load_model():
    # state = create_train_state(rng, config, vocab_size)
    checkpoint_path = '/home/abhi98m/backed_up/taylordiff/experiments/model_checkpoints/transformer/epoch_0'
    orbax_checkpointer =  orbax.checkpoint.PyTreeCheckpointer()
    weights =  orbax_checkpointer.restore(checkpoint_path)
    return weights["params"]


def test_attention_maps(model_config):
    model = TransformerModel(**model_config)
    
    # Create a sample input
    batch_size, seq_length = 1, 10
    x = jnp.arange(seq_length)[None, :]  # Create a sequence of integers
    
    # Initialize the model
    key = jax.random.PRNGKey(jax.random.randint(jax.random.PRNGKey(0), (), minval=0, maxval=2**10-1))
    params = model.init(key, x, training=False)
    
    # Get model output and attention weights
    _, attention_maps = model.apply(params, x, training=False)
    attention_weights = attention_maps
    print(attention_weights)
    # print("Shape of attention maps:",attention_weights.shape) 
    print(len(attention_weights))
    print("Shape of attention maps:",attention_weights[0].shape) 
    print("Shape of attention maps:",attention_weights[0][0,0,:,:].shape) 
    # Plot attention maps
    fig, axes = plt.subplots(model_config['n_layers'], model_config['n_heads'], 
                             figsize=(20, 5 * model_config['n_layers']))
    
    for layer in range(model_config['n_layers']):
        for head in range(model_config['n_heads']):
            ax = axes[layer, head]
            sns.heatmap(attention_weights[layer][0,head,:,:], ax=ax, cmap='inferno', vmin=0, vmax=0.99)
            ax.set_title(f'Layer {layer+1}, Head {head+1}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
    
    plt.tight_layout()
    plt.savefig('attention_maps.png')
    plt.close()

    # Assert that the attention maps are causal (upper triangular)
    for layer_attn in attention_weights:
        for head_attn in layer_attn[0]:  # [0] to get the first (and only) batch
            assert jnp.allclose(jnp.triu(head_attn), head_attn), "Attention map is not causal (upper triangular)"

    # Assert that the attention weights sum to 1 along the key dimension
    for layer_attn in attention_weights:
        assert jnp.allclose(layer_attn.sum(axis=-1), 1.0), "Attention weights do not sum to 1"