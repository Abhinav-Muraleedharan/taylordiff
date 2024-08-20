# Example of creating a simple checkpoint
import os
import orbax.checkpoint
import jax.numpy as jnp
import jax
from models import transformer

# Define a simple checkpoint directory
checkpoint_path = '/home/abhi98m/backed_up/taylordiff/experiments/model_checkpoints/transformer/epoch_0'
# checkpoint_path = os.path.join(checkpoint_dir, 'simple_checkpoint.chkpt')

# Initialize Orbax CheckpointManager
orbax_checkpointer =  orbax.checkpoint.PyTreeCheckpointer()

# Define a simple state
simple_state = [1, {"k1": 2, "k2": (3, 4)}, 5]


# # Save the checkpoint
# orbax_checkpointer.save(checkpoint_path, simple_state)

# Load the checkpoint
print(dir(orbax_checkpointer))
loaded_state = orbax_checkpointer.restore(checkpoint_path)
print(loaded_state)

model = transformer.TransformerModel()

model.apply(loaded_state,1)