import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
import wandb

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
    params = model.init(rng, jnp.ones((1, config['training']['max_seq_length'])), training=False)['params']
    tx = optax.adam(config['training']['learning_rate'])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_step(state, batch, rng):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input_ids'], training=True, rngs={'dropout': rng})
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch['input_ids']).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['input_ids'], training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['input_ids']).mean()
    return loss

def train_model(config, train_dataset, val_dataset, vocab_size):
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, config, vocab_size)

    train_losses = []
    val_losses = []

    fixed_input = next(train_dataset.iter(batch_size=1))['input_ids']
    attention_weights_history = []

    for epoch in range(config['training']['num_epochs']):
        # Training
        train_loss = 0
        for batch in tqdm(train_dataset.iter(batch_size=config['training']['batch_size']), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}"):
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(state, batch, step_rng)
            train_loss += loss

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

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        _, attention_weights = state.apply_fn({'params': state.params}, fixed_input, training=False, return_attention=True)
        attention_weights_history.append([w.numpy() for w in attention_weights])

        log_attention_heatmaps(attention_weights, epoch)

    plot_loss_curves(train_losses, val_losses)
    plot_spectral_dynamics(attention_weights_history)

    return state