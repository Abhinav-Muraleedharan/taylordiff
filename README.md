

## Setup

1. **Create and activate the Conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate taylordiff
    ```

2. **Run the main script:**

    ```bash
    python main.py
    ```

## Overview

- **`config/`**: Contains configuration files.
- **`src/`**: Source code files.
  - `__init__.py`: Initialization script for the source module.
  - `data.py`: Handles data processing.
  - `train.py`: Contains training routines.
  - `utils.py`: Utility functions.
- **`models/`**: Model definitions.
  - `__init__.py`: Initialization script for the models module.
  - `transformer.py`: Transformer model implementation.
  - `gpt2.py`: GPT-2 model implementation.
  - `gemma.py`: Gemma model implementation.
- **`requirements.txt`**: List of Python packages required for the project.
- **`environment.yml`**: Conda environment configuration.
- **`README.md`**: This file.
- **`main.py`**: Main entry point of the application.


