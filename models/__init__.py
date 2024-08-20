from .transformer import TransformerModel
from .frozen_transformer import FrozenTransformerModel
from .gpt2 import GPT2Model
from .gemma import GemmaModel

MODEL_REGISTRY = {
    "frozen_transformer": FrozenTransformerModel,
    "transformer": TransformerModel,
    "gpt2": GPT2Model,
    "gemma": GemmaModel,
}

def get_model(model_type, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](**kwargs)