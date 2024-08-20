
import os
import orbax.checkpoint

orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.ArrayCheckpointHandler())

checkpoint_path = '/home/abhi98m/backed_up/taylordiff/experiments/model_checkpoints/transformer/epoch_0'

state = orbax_checkpointer.restore(checkpoint_path)
print(state)