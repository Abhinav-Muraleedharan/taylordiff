from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_data(config):
    dataset = load_dataset(config['data']['dataset_name'], config['data']['dataset_config'])
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=config['training']['max_seq_length'], padding='max_length')

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_dataset = tokenized_dataset.with_format("jax")

    train_dataset = tokenized_dataset['train']
    val_dataset = tokenized_dataset['validation']

    return train_dataset, val_dataset, tokenizer.vocab_size