from pathlib import Path
from tqdm import tqdm

import argparse
import pandas as pd
import torch

from common import read_yaml
from utils import load_checkpoint
from utils import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='settings.yaml', help='config file (default: settings.yaml)')
    parser.add_argument('--ckpt', help='model checkpoint to use', default=None)
    parser.add_argument('caption_file')
    parser.add_argument('-o', '--output', default='predictions.csv', help='output filename')
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load captions
    capt_fn = Path(args.caption_file)
    if capt_fn.suffix == '.txt':
        captions = capt_fn.read_text().splitlines()
    elif capt_fn.suffix == '.csv':
        df = pd.read_csv(capt_fn)
        captions = list(df['caption'])
    
    ckpt_path = args.ckpt if args.ckpt is not None else cfg.get('model_path')
    tokenizer = load_tokenizer(cfg.get('tokenizer_dir'))
    model = load_checkpoint(ckpt_path)
    model.to(device)

    rows = []
    for caption in tqdm(captions):
        tokens = tokenizer.encode(caption).ids
        tokens = torch.as_tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)
        pred, _ = model(tokens)
        pred = torch.softmax(pred[0].cpu(), dim=-1).numpy()

        m = 'audio' if pred[0] > pred[1] else 'image'
        
        rows.append([caption, m, pred[0], pred[1]])

    df = pd.DataFrame(rows, columns=['caption', 'modality', 'p_audio', 'p_image']).set_index('caption')
    df.to_csv(args.output)


if __name__ == '__main__':
    main()
