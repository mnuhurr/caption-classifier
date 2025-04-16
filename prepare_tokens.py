from pathlib import Path

from common import read_yaml, write_json, init_log
from utils import load_tokenizer


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('tokenize')

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    cache_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = load_tokenizer(cfg.get('tokenizer_dir'))

    for split in ['train', 'val']:
        tokens = {}
        for modality, fn_list in cfg.get(f'{split}_captions', {}).items():
            if modality not in tokens:
                tokens[modality] = []

            for fn in fn_list:
                captions = Path(fn).read_text().splitlines()
                file_tokens = [t.ids for t in tokenizer.encode_batch(captions)]
                tokens[modality].extend(file_tokens)

        n_img = len(tokens['image'])
        n_aud = len(tokens['audio'])
        logger.info(f'{split}: {n_img} captions for images, {n_aud} captions for audio')
        write_json(cache_dir / f'tokens_{split}.json', tokens)


if __name__ == '__main__':
    main()
