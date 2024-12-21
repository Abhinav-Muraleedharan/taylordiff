# Code Examples

## Basic Usage

### 1. Training a Model

```python
from taylordiff.models import Transformer
from taylordiff.src.train import Trainer
from taylordiff.src.data import DataModule
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model and data
model = Transformer(
    vocab_size=config['model']['vocab_size'],
    d_model=config['model']['d_model'],
    nhead=config['model']['nhead'],
    num_layers=config['model']['num_layers']
)

data_module = DataModule(
    train_path=config['data']['train_path'],
    val_path=config['data']['val_path'],
    batch_size=config['training']['batch_size']
)

# Initialize trainer
trainer = Trainer(
    model=model,
    config=config['training']
)

# Train the model
trainer.train(data_module)
```

### 2. Analyzing Attention Patterns

```python
from taylordiff.analysis.generate import AttentionAnalyzer
import torch

# Load a trained model
model_path = "path/to/checkpoint.pt"
model = Transformer.load_from_checkpoint(model_path)

# Initialize analyzer
analyzer = AttentionAnalyzer(model)

# Generate attention maps
input_text = "Example input for analysis"
attention_maps = analyzer.generate_attention_maps(input_text)

# Visualize specific layer's attention
analyzer.plot_attention_head(
    attention_maps,
    layer=0,
    head=0,
    save_path="attention_maps.png"
)
```

### 3. Circuit Analysis

```python
from taylordiff.analysis import CircuitAnalyzer
import torch

# Initialize circuit analyzer
analyzer = CircuitAnalyzer(model)

# Identify and analyze emergent circuits
circuits = analyzer.find_circuits(
    threshold=0.5,  # Activation threshold
    min_connections=3  # Minimum connections to form a circuit
)

# Visualize circuit
analyzer.plot_circuit(
    circuit_id=0,
    save_path="circuit_visualization.png"
)
```

### 4. Custom Training Loop with Model Approximations

```python
from taylordiff.approximations import TaylorApproximation
from taylordiff.models import FrozenTransformer
import torch

# Initialize models
base_model = Transformer(vocab_size=50257, d_model=768)
frozen_model = FrozenTransformer(vocab_size=50257, d_model=768)

# Initialize approximation
approximation = TaylorApproximation(
    base_model=base_model,
    order=2  # Second-order approximation
)

# Training loop with approximation
optimizer = torch.optim.AdamW(base_model.parameters())

for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass with approximation
        output = approximation(batch)
        loss = criterion(output, batch['labels'])
        
        # Compare with frozen model
        frozen_output = frozen_model(batch)
        frozen_loss = criterion(frozen_output, batch['labels'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Experiment Configuration

Example `config.yaml`:

```yaml
model:
  type: "transformer"
  vocab_size: 50257
  d_model: 768
  nhead: 12
  num_layers: 12
  dropout: 0.1

data:
  train_path: "data/train.txt"
  val_path: "data/val.txt"
  max_length: 1024
  
training:
  batch_size: 32
  learning_rate: 1e-4
  max_epochs: 10
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  
wandb:
  project: "taylordiff"
  entity: "your-username"
  
approximation:
  type: "taylor"
  order: 2
  freeze_attention: false
```

### 6. Using the Command Line Interface

Train a model:
```bash
python main.py train \
    --config config/config.yaml \
    --model transformer \
    --checkpoint-dir checkpoints/
```

Generate attention analysis:
```bash
python analysis/generate.py \
    --model-path checkpoints/latest.ckpt \
    --input-text "Example text for analysis" \
    --output-dir outputs/attention_maps/
```

Run circuit analysis:
```bash
python analysis/circuits.py \
    --model-path checkpoints/latest.ckpt \
    --threshold 0.5 \
    --output-dir outputs/circuits/
```
