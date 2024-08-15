import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wandb

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TaylorDiff: Training and Validation Loss Curves')
    plt.legend()
    
    wandb.log({"loss_curves": wandb.Image(plt)})
    
    plt.close()

def plot_spectral_dynamics(attention_weights_history):
    n_layers = len(attention_weights_history[0])
    n_heads = attention_weights_history[0][0].shape[0]
    n_epochs = len(attention_weights_history)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4*n_heads, 4*n_layers), squeeze=False)
    
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            
            attn_weights = [epoch_weights[layer][head] for epoch_weights in attention_weights_history]
            
            singular_values = [np.linalg.svd(weights, compute_uv=False) for weights in attn_weights]
            
            im = ax.imshow(singular_values, aspect='auto', cmap='viridis')
            ax.set_title(f'Layer {layer+1}, Head {head+1}')
            ax.set_xlabel('Singular Value Index')
            ax.set_ylabel('Epoch')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    plt.suptitle('TaylorDiff: Spectral Dynamics of Attention Patterns')
    plt.tight_layout()
    
    wandb.log({"spectral_dynamics": wandb.Image(fig)})
    
    plt.close()

def log_attention_heatmaps(attention_weights, epoch):
    n_layers = len(attention_weights)
    n_heads = attention_weights[0].shape[0]

    for layer in range(n_layers):
        for head in range(n_heads):
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(attention_weights[layer][head], cmap='viridis')
            ax.set_title(f'TaylorDiff: Layer {layer+1}, Head {head+1}')
            ax.set_xlabel('Token (key)')
            ax.set_ylabel('Token (query)')
            plt.colorbar(im)
            
            wandb.log({f"attention_heatmap_layer{layer+1}_head{head+1}": wandb.Image(fig)}, step=epoch)
            
            plt.close()