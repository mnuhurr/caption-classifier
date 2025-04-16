from pathlib import Path

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import BertNormalizer


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


