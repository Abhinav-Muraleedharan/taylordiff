# TaylorDiff: Analyzing LLM Finetuning Dynamics and Emergent Circuits

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://your-documentation-url.com)

TaylorDiff is a comprehensive toolkit for analyzing the dynamics of Large Language Model finetuning and investigating emergent circuits formed during the training process. This project provides tools and methodologies for understanding how neural networks evolve during training and how different architectural components interact to form functional circuits.

## Key Features

- **Finetuning Analysis**: Track and visualize how model parameters evolve during training
- **Circuit Detection**: Identify and analyze emergent circuits in transformer models
- **Model Approximations**: Study model behavior using various approximation techniques
- **Attention Analysis**: Visualize and analyze relavant attention patterns across different layers
- **Multiple Architectures**: Support for Transformer, GPT-2, and Gemma architectures
- **Experiment Tracking**: Integration with Weights & Biases for comprehensive experiment logging

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/taylordiff.git
   cd taylordiff
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate taylordiff
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

## Quick Start

### Basic Training
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

# Train model
trainer = Trainer(model=model, config=config['training'])
trainer.train(DataModule(config['data']))
```

### Analyzing Attention Patterns
```python
from taylordiff.analysis.generate import AttentionAnalyzer

analyzer = AttentionAnalyzer(model)
attention_maps = analyzer.generate_attention_maps("Example input")
analyzer.plot_attention_head(attention_maps, layer=0, head=0)
```

## Project Structure

```
.
├── analysis/               # Analysis scripts
│   ├── __init__.py
│   └── generate.py        # Attention map generation
├── approximations/        # Model approximation implementations
├── config/               # Configuration files
│   ├── config.yaml
│   └── attention_frozen_config.yaml
├── models/               # Model architectures
│   ├── transformer.py
│   ├── gpt2.py
│   └── gemma.py
├── src/                 # Core functionality
│   ├── data.py         # Data processing
│   ├── train.py        # Training routines
│   └── utils.py        # Utility functions
└── tests/              # Test suite
```

## Detailed Usage

### 1. Configuration

Create a configuration file (`config.yaml`):
```yaml
model:
  type: "transformer"
  vocab_size: 50257
  d_model: 768
  nhead: 12
  num_layers: 12
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 1e-4
  max_epochs: 10
  gradient_clip_val: 1.0
```

### 2. Circuit Analysis

```python
from taylordiff.analysis import CircuitAnalyzer

analyzer = CircuitAnalyzer(model)
circuits = analyzer.find_circuits(threshold=0.5)
analyzer.plot_circuit(circuit_id=0)
```

### 3. Command Line Interface

Train a model:
```bash
python main.py train \
    --config config/config.yaml \
    --model transformer \
    --checkpoint-dir checkpoints/
```

Generate analysis:
```bash
python analysis/generate.py \
    --model-path checkpoints/latest.ckpt \
    --output-dir outputs/
```

## Experiment Tracking

TaylorDiff uses Weights & Biases for experiment tracking. View your runs:
```bash
wandb sync wandb/run-*
```

Key metrics tracked:
- Training/validation loss
- Attention pattern evolution
- Circuit formation dynamics
- Parameter distribution changes
- Approximation accuracy

## Advanced Features

### Model Approximations
```python
from taylordiff.approximations import TaylorApproximation

approximation = TaylorApproximation(base_model=model, order=2)
output = approximation(input_data)
```

### Custom Training Loops
```python
from taylordiff.models import FrozenTransformer
import torch

base_model = Transformer(vocab_size=50257, d_model=768)
frozen_model = FrozenTransformer(vocab_size=50257, d_model=768)

optimizer = torch.optim.AdamW(base_model.parameters())

for epoch in range(num_epochs):
    for batch in dataloader:
        output = base_model(batch)
        loss = criterion(output, batch['labels'])
        loss.backward()
        optimizer.step()
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Before submitting a PR:
- Ensure tests pass (`python -m pytest tests/`)
- Update documentation if needed
- Follow our coding standards

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
