import torch


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''

    def __init__(self, config, pre_seq_len, prefix_hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(pre_seq_len, config.hidden_size)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, prefix_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
        )

    def forward(self, prefix: torch.Tensor):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values
