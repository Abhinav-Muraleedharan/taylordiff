import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import orbax.checkpoint
from transformers import AutoTokenizer
import yaml
from models import get_model
from src.data import load_and_preprocess_data




def create_model(config, vocab_size):
    model = get_model(
        config['model']['type'],
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout']
    )
    return model

def create_train_state(rng, config, vocab_size):
    model = create_model(config, vocab_size)
    params = model.init(rng, jnp.ones((1, config['training']['max_seq_length']), dtype=jnp.int32), training=False)['params']
    tx = optax.adam(config['training']['learning_rate'])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def load_model(config, vocab_size):
    # state = create_train_state(rng, config, vocab_size)
    model_name = config['model']['type']
    checkpoint_path = '/home/abhi98m/backed_up/taylordiff/experiments/model_checkpoints/transformer/epoch_0'
    orbax_checkpointer =  orbax.checkpoint.PyTreeCheckpointer()
    params =  orbax_checkpointer.restore(checkpoint_path)
    # print(params)
    return params

def generate(params, config, tokenizer, prompt, max_length=20):
    model = create_model(config, tokenizer.vocab_size)
    
    @jax.jit
    def predict_next_token(params, input_ids):
        logits = model.apply({'params': params}, input_ids, training=False)
        return logits[:, -1, :]

    input_ids = tokenizer.encode(prompt, return_tensors='jax').squeeze()
    
    for _ in range(max_length):
        logits = predict_next_token(params, input_ids[None, :])
        next_token = jnp.argmax(logits, axis=-1)
        input_ids = jnp.concatenate([input_ids, next_token], axis=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids)

def main():

    
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])
    _, _, vocab_size = load_and_preprocess_data(config)
    params = load_model(config, vocab_size)
    # print(params)
    prompt = "Please note restoration time will"
    generated_text = generate(params['params'], config, tokenizer, prompt)
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()