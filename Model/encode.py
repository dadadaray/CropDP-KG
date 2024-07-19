import torch
import torch.nn as nn

from config import PromptConfig


class PrefixEncoder(nn.Module):
    """
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """
    def __init__(self):
        super().__init__()
        self.prefix_projection = PromptConfig.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(PromptConfig.pre_seq_len, PromptConfig.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(PromptConfig.hidden_size, PromptConfig.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(PromptConfig.prefix_hidden_size, PromptConfig.num_hidden_layers * 2 * PromptConfig.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(PromptConfig.pre_seq_len, PromptConfig.num_hidden_layers * 2 * PromptConfig.hidden_size)

    def forward(self, prefix: torch.Tensor):            # prefix:[batch_size, pre_seq_len]
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class PrefixEncoderI(nn.Module):
    """
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """
    def __init__(self):
        super().__init__()
        self.prefix_projection = PromptConfig.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(PromptConfig.pre_seq_len, PromptConfig.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(PromptConfig.hidden_size, PromptConfig.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(PromptConfig.prefix_hidden_size, PromptConfig.independent_layers * 2 * PromptConfig.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(PromptConfig.pre_seq_len, PromptConfig.independent_layers * 2 * PromptConfig.hidden_size)

    def forward(self, prefix: torch.Tensor):            # prefix:[batch_size, pre_seq_len]
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class PrefixEncoderS(nn.Module):
    """
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """
    def __init__(self):
        super().__init__()
        self.prefix_projection = PromptConfig.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(PromptConfig.pre_seq_len, PromptConfig.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(PromptConfig.hidden_size, PromptConfig.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(PromptConfig.prefix_hidden_size, PromptConfig.shared_layers * 2 * PromptConfig.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(PromptConfig.pre_seq_len, PromptConfig.shared_layers * 2 * PromptConfig.hidden_size)

    def forward(self, prefix: torch.Tensor):            # prefix:[batch_size, pre_seq_len]
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


