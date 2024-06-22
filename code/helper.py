import torch
import torch.nn as nn
import requests
from PIL import Image

import warnings

# Ignore specific warnings related to max_length in transformers
warnings.filterwarnings(action="ignore", message=".*Using the model-agnostic default `max_length`.*")

class DummyModel(nn.Module):
    """
    A dummy model that consists of an embedding layer
    with two blocks of a linear layer followed by a norm layer.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        torch.manual_seed(123)

        self.token_embedding = nn.Embedding(num_embeddings=2, embedding_dim=2)

        # Block 1
        self.linear_1 = nn.Linear(in_features=2, out_features=2)
        self.layernorm_1 = nn.LayerNorm(normalized_shape=2)

        # Block 2
        self.linear_2 = nn.Linear(in_features=2, out_features=2)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=2)

        self.head = nn.Linear(in_features=2, out_features=2)

    def forward(self, x):
        hidden_states = self.token_embedding(x)

        # Block 1
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.layernorm_1(hidden_states)

        # Block 2
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.layernorm_2(hidden_states)

        logits = self.head(hidden_states)

        return logits

def get_generation(model, processor, image, dtype):
    inputs = processor(image, return_tensors="pt").to(dtype)
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)

def load_image(img_url):
    image = Image.open(requests.get(url=img_url, stream=True).raw).convert("RGB")

    return image
