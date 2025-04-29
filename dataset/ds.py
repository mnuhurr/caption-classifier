
import torch
import numpy as np


def collate_tokens(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    bs = len(batch)
    max_len = max(map(len, batch))

    x = torch.zeros(bs, max_len, dtype=torch.int64)
    mask = torch.ones(bs, max_len, dtype=bool)

    for k, tokens in enumerate(batch):
        tlen = len(tokens)
        x[k, :tlen] = tokens
        mask[k, :tlen] = False

    return x, mask


def collate_batch(batch):
    tokens, tags = list(zip(*batch))
    x, x_mask = collate_tokens(tokens)
    y = torch.stack(tags)

    return x, x_mask, y


class RandomSamplingDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, n_samples_in_epoch: int = 100000, seed: int | None = None):
        self.tokens = tokens
        self.modalities = sorted(self.tokens.keys())
        self.n_samples = n_samples_in_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        m_idx = idx % len(self.modalities)
        m = self.modalities[m_idx]

        nc = len(self.tokens[m])
        k = self.rng.integers(nc)
        tokens = torch.as_tensor(self.tokens[m][k], dtype=torch.int64)
        return tokens, torch.as_tensor(m_idx, dtype=torch.int64)


class BalancedSamplingDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, n_samples_in_epoch: int = 100000, seed: int | None = None):
        self.tokens = tokens
        self.modalities = sorted(self.tokens.keys())
        self.n_samples = n_samples_in_epoch
        self.rng = np.random.default_rng(seed)
    
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        m_idx = idx % len(self.modalities)
        m = self.modalities[m_idx]

        idx = idx // len(self.modalities)
        n_idx = idx % len(self.tokens[m])
        n = sorted(self.tokens[m].keys())[n_idx]

        nc = len(self.tokens[m][n])
        k = self.rng.integers(nc)
        tokens = torch.as_tensor(self.tokens[m][n][k], dtype=torch.int64)
        return tokens, torch.as_tensor(m_idx, dtype=torch.int64)


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens):
        self.data = []

        mods = sorted(tokens.keys())
        for m_idx, m in enumerate(mods):
            for caption_tokens in tokens[m]:
                caption_tokens = torch.as_tensor(caption_tokens, dtype=torch.int64)
                m_id = torch.as_tensor(m_idx, dtype=torch.int64)
                self.data.append((caption_tokens, m_id))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


