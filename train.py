
from pathlib import Path
import argparse

import time
import torch
import torch.nn.functional as F

from common import read_json, read_yaml, init_log
from dataset import RandomSamplingDataset, TokenDataset, collate_batch
#from models.classifier import AttentionClassifierConfig, AttentionClassifier
from models.classifier import ClassifierConfig, TransformerClassifier
#from models.classifier import LSTMClassifierConfig, LSTMClassifier
from models.utils import model_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='settings.yaml', help='config file (default: settings.yaml)')
    return parser.parse_args()


def train_epoch(model: torch.nn.Module,
                loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None,
                clip_grad_norm: float | None = None,
                log_interval: int | None = None,
                device: torch.device | None = None) -> float:
    model.train()
    batch_t0 = time.perf_counter()
    train_loss = 0.0

    scaler = torch.amp.GradScaler(device)
    device_type = device.type if device is not None else 'cpu'

    for batch, (x, x_mask, y_true) in enumerate(loader):
        x = x.to(device)
        x_mask = x_mask.to(device)
        y_true = y_true.to(device)

        with torch.amp.autocast(device_type=device_type):
            y_pred, _ = model(x, mask=x_mask)
            loss = F.cross_entropy(y_pred, y_true)

        train_loss += loss.item()
        scaler.scale(loss).backward()

        if clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        if log_interval is not None and batch % log_interval == 0:
            t_batch = int(1000 * (time.perf_counter() - batch_t0) / log_interval)
            lr = optimizer.param_groups[0]['lr']

            print(f'batch {batch:4d}/{len(loader)} - {t_batch} ms/batch - lr {lr:.4g} - training loss {loss.item():.4f}')
            batch_t0 = time.perf_counter()

    return train_loss / len(loader)


@torch.inference_mode()
def validate_epoch(model, loader, device):
    model.eval()
    val_loss = 0.0

    confusion = torch.zeros(2, 2)

    for x, x_mask, y_true in loader:
        x = x.to(device)
        x_mask = x_mask.to(device)
        y_true = y_true.to(device)
        y_pred, _ = model(x, mask=x_mask)
        val_loss += F.cross_entropy(y_pred, y_true).item()

        for yp, yt in zip(torch.argmax(y_pred, dim=-1), y_true):
            confusion[yt, yp] += 1

    confusion = confusion / torch.sum(confusion, dim=1, keepdim=True)

    return val_loss / len(loader), confusion


def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    logger = init_log('train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cache_dir = Path(cfg.get('cache_dir', 'cache'))

    model_path = Path(cfg.get('model_path', 'model-ckpt.pt'))
    model_path.parent.mkdir(exist_ok=True, parents=True)
    
    train_tokens = read_json(cache_dir / 'tokens_train.json')
    val_tokens = read_json(cache_dir / 'tokens_val.json')

    epoch_len = cfg.get('epoch_len', 100_000)
    batch_size = cfg.get('batch_size', 64)
    num_workers = cfg.get('num_dataloader_workers', 8)
    n_epochs = cfg.get('n_epochs', 10)
    warmup_pct = cfg.get('warmup_pct', 0.0)

    learning_rate = cfg.get('learning_rate', 1e-3)
    
    clip_grad_norm = cfg.get('clip_grad_norm')
    
    log_interval = cfg.get('log_interval')

    ds_train = RandomSamplingDataset(train_tokens, n_samples_in_epoch=epoch_len)
    ds_val = TokenDataset(val_tokens)

    # no need to shuffle: the training dataset is sampling
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_batch)

    """
    model_cfg = AttentionClassifierConfig(
        vocab_size=cfg.get('vocab_size', 10000),
        d_model=cfg.get('d_model', 256),
        dropout=cfg.get('dropout', 0.0),
        p_masking=cfg.get('p_masking', 0.0),
        mask_token_id=cfg.get('mask_token_id', 4))
    """
    model_cfg = ClassifierConfig(
        vocab_size=cfg.get('vocab_size', 10000),
        d_model=cfg.get('d_model', 256),
        n_heads=cfg.get('n_heads', 8),
        n_layers=cfg.get('n_layers', 4),
        dropout=cfg.get('dropout', 0.0),
        p_masking=cfg.get('p_masking', 0.0),
        mask_token_id=cfg.get('mask_token_id', 4))

    """
    model_cfg = LSTMClassifierConfig(
        vocab_size=cfg.get('vocab_size', 10000),
        d_embedding=cfg.get('d_embedding', 64),
        d_lstm=cfg.get('d_lstm', 128),
        n_layers=cfg.get('n_layers', 2),
        dropout=cfg.get('dropout', 0.0))
    """

    print(model_cfg)
    #model = AttentionClassifier(model_cfg)
    #model = LSTMClassifier(model_cfg)
    model = TransformerClassifier(model_cfg)
    
    logger.info(f'model size {model_size(model) / 1e6:.1f}M')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        total_steps=n_epochs * len(train_loader),
        max_lr=learning_rate,
        pct_start=warmup_pct)

    #best_loss = float('inf')
    best_score = 0.0

    for epoch in range(n_epochs):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            clip_grad_norm=clip_grad_norm,
            log_interval=log_interval,
            device=device)

        val_loss, confmat = validate_epoch(model, val_loader, device=device)

        logger.info(f'epoch {epoch + 1} - training loss {train_loss:.4f} - validation loss {val_loss:.4f}')
        print(confmat)

        #val_conf_loss = confmat[0, 1] + confmat[1, 0]
        val_score = confmat[0, 0] * confmat[1, 1]

        #if val_conf_loss < best_loss:
        if val_score > best_score:
            #best_loss = val_conf_loss
            best_score = val_score
            ckpt = {
                'config': model_cfg,
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }

            torch.save(ckpt, model_path)


if __name__ == '__main__':
    main()

