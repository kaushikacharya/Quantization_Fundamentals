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

# Monkey patching for quanto
def named_module_tensors(module, recurse=False):
    for named_parameter in module.named_parameters(recurse=recurse):
        name, val = named_parameter
        flag = True
        if hasattr(val, "_data") or hasattr(val, "_scale"):
            if hasattr(val, "_data"):
                yield name + "._data", val._data
            if hasattr(val, "_scale"):
                yield name + "._scale", val._scale
        else:
            yield named_parameter

    for named_buffer in module.named_buffers(recurse=recurse):
        yield named_buffer

def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    """
    import re
    
    if dtype == torch.bool:
        return 1/8
    
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid data type: {dtype}.")
    
    bit_size = int(bit_search.groups()[0])
    return bit_size / 8

def compute_model_sizes(model):
    """
    Compute the size of each submodule of a given model
    """
    from collections import defaultdict

    module_sizes = defaultdict(int)

    for name, tensor in named_module_tensors(module=model, recurse=True):
        size = tensor.numel() * dtype_byte_size(tensor.dtype)
        name_parts = name.split(".")
        for idx in range(len(name_parts)+1):
            module_sizes[".".join(name_parts[:idx])] += size
    
    return module_sizes
