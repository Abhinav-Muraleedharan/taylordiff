import os
import jax
import optax
import wandb
import orbax.checkpoint
import jax.numpy as jnp
from tqdm import tqdm
from flax.training import checkpoints
from flax.training import train_state
from models import get_model
from .utils import plot_loss_curves, plot_spectral_dynamics, log_attention_heatmaps


def create_train_state(rng, config, vocab_size):
    model = get_model(
        config['model']['type'],
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout']
    )
    dummy_input = jnp.ones((1, config['training']['max_seq_length']), dtype=jnp.int32)
    params = model.init(rng, dummy_input, training=False)['params']
    tx = optax.adam(config['training']['learning_rate'])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_step(state, batch, rng):
    def loss_fn(params):
        logits,_ = state.apply_fn({'params': params}, batch['input_ids'], training=True, rngs={'dropout': rng})
        shifted_logits = logits[:, :-1, :]
        shifted_targets = batch['input_ids'][:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(shifted_logits, shifted_targets).mean()
        return  loss 

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def eval_step(state, batch):
    logits,_  = state.apply_fn({'params': state.params}, batch['input_ids'], training=False)
    shifted_logits = logits[:, :-1, :]
    shifted_targets = batch['input_ids'][:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(shifted_logits, shifted_targets).mean()
    return loss

def train_model(config, train_dataset, val_dataset, vocab_size):
    save_checkpoints = [0,20]
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, config, vocab_size)
    model_name = config['model']['type']
    checkpoint_dir = os.path.abspath(os.path.join('experiments', 'model_checkpoints', model_name))
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(config['training']['num_epochs']):
        # Training
        train_loss = 0
        for batch in tqdm(train_dataset.iter(batch_size=config['training']['batch_size']), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}"):
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(state, batch, step_rng)
            train_loss += loss
            print(train_loss)

        train_loss /= len(train_dataset) // config['training']['batch_size']
        train_losses.append(train_loss)

        # Validation
        val_loss = 0
        for batch in val_dataset.iter(batch_size=config['training']['batch_size']):
            loss = eval_step(state, batch)
            val_loss += loss

        val_loss /= len(val_dataset) // config['training']['batch_size']
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # save checkpoint:
        if epoch in save_checkpoints:
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            orbax_checkpointer.save(os.path.join(checkpoint_dir,f'epoch_{epoch}'),state)
            print("Succesfully saved checkpoint")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    plot_loss_curves(train_losses, val_losses)

    return state