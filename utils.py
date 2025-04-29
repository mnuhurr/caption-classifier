from pathlib import Path

import torch

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import BertNormalizer

from models.classifier import AttentionClassifierConfig, AttentionClassifier
from models.classifier import LSTMClassifierConfig, LSTMClassifier
from models.classifier import ClassifierConfig, TransformerClassifier
from models.classifier import ProjectorConfig, ProjectorClassifier


def load_tokenizer(tokenizer_path, max_length=None):
    vocab_fn = str(Path(tokenizer_path) / 'tokenizer-vocab.json')
    merges_fn = str(Path(tokenizer_path) / 'tokenizer-merges.txt')

    tokenizer = ByteLevelBPETokenizer(vocab_fn, merges_fn)
    tokenizer._tokenizer.post_processor = BertProcessing(
        ('</s>', tokenizer.token_to_id('</s>')),
        ('<s>', tokenizer.token_to_id('<s>')),
    )

    tokenizer._tokenizer.normalizer = BertNormalizer(strip_accents=False, lowercase=True)

    if max_length is not None:
        tokenizer.enable_truncation(max_length=max_length)

    return tokenizer



def load_checkpoint(filename: str | Path) -> torch.nn.Module:
    ckpt = torch.load(filename, map_location='cpu')

    cfg = ckpt['config']
    print(cfg)
    if 'epoch' in ckpt:
        print('model saved on epoch {}'.format(ckpt['epoch']))

    if isinstance(cfg, AttentionClassifierConfig):
        model = AttentionClassifier(cfg)
    elif isinstance(cfg, LSTMClassifierConfig):
        model = LSTMClassifier(cfg)
    elif isinstance(cfg, ClassifierConfig):
        model = TransformerClassifier(cfg)
    elif isinstance(cfg, ProjectorConfig):
        model = ProjectorClassifier(cfg)

    model.load_state_dict(ckpt['state_dict'])

    return model

