
## Running
1. fit tokenizer (`python fit_tokenizer.py`)
2. prepare tokens (`python prepare_tokens.py`)
3. train the model (`python train.py`)
4. run the app (`python run flaskapp.py`)


### example `settings.yaml`:
```
train_captions:
  image:
    - data/coco/captions_train2014.txt

  audio:
    - data/clotho/clotho_captions_development.txt
    - data/audiocaps2.0/train.txt
    - data/macs/captions-macs-bias.txt
    - data/macs/captions-macs-nobias.txt

val_captions:
  image:
    - data/coco/captions_val2014.txt

  audio:
    - data/clotho/clotho_captions_validation.txt
    - data/audiocaps2.0/val.txt

min_frequency: 2
vocab_size: 20000
tokenizer_dir: data/tokenizer

cache_dir: data/cache

model_path: data/model.pt

batch_size: 1024
num_dataloader_workers: 8

epoch_len: 400000
n_epochs: 10
warmup_pct: 0.05
learning_rate: 1.0e-2
clip_grad_norm: 1.0
log_interval: 100

d_model: 256
n_heads: 8
n_layers: 1
dropout: 0.2
p_masking: 0.50
```
