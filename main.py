import yaml
import wandb
from src.data import load_and_preprocess_data
from src.train import train_model

def main():
    with open('config/attention_frozen_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    wandb.init(project=config['project']['name'], config=config)

    train_dataset, val_dataset, vocab_size = load_and_preprocess_data(config)
    trained_state = train_model(config, train_dataset, val_dataset, vocab_size)

    wandb.finish()

if __name__ == "__main__":
    main()