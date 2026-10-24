"""Edit CONFIG below, then run: python train_single_category.py."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from train_angel import AngelDataset, AngelModel
from main import SketchDiffusion
from data.dm import ReprType
from data.batching import ThreePointAbs


# 在这里填写训练参数。只使用训练划分；不会读取 valid/test。
CONFIG = {
    'npz_path': r'D:\document\DataSet\quickdraw\raw\angel.full.npz',
    'batch_size': 128,
    'random_seed': 5555,
    'epoch': 200,
    'num_images': 16,
}

PROJECT_ROOT = Path(__file__).resolve().parent
CHINA_TIME = timezone(timedelta(hours=8))


def save_image(points, length, path):
    sample = points[:int(length)]
    fig, ax = plt.subplots(figsize=(4, 4))
    for stroke in np.split(sample[:, :2], np.flatnonzero(sample[:, 2] > .8) + 1):
        if len(stroke) > 1:
            ax.plot(stroke[:, 0], stroke[:, 1], color='black', linewidth=1.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    fig.savefig(path, dpi=128, facecolor='white', bbox_inches='tight', pad_inches=.1)
    plt.close(fig)


class EpochProgress(pl.Callback):
    def on_train_epoch_end(self, trainer, model):
        print(f'Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}, '
              f'step={trainer.global_step}', flush=True)


def train_and_generate(npz_path, batch_size, random_seed, epoch, num_images):
    for name, value in [('batch_size', batch_size), ('epoch', epoch), ('num_images', num_images)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not isinstance(random_seed, int) or isinstance(random_seed, bool) or not 0 <= random_seed < 2**32:
        raise ValueError('random_seed must be an integer in [0, 2**32)')
    data_path = Path(npz_path).expanduser().resolve()
    if not data_path.is_file() or data_path.suffix.lower() != '.npz':
        raise ValueError(f'NPZ file does not exist: {data_path}')
    category = data_path.stem.removesuffix('.full')
    if not category or category in ('.', '..'):
        raise ValueError('Cannot extract category from the dataset filename')
    if not torch.cuda.is_available():
        raise RuntimeError('This training script requires a CUDA GPU')

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(random_seed, workers=True)
    with np.load(data_path, allow_pickle=True, encoding='latin1') as archive:
        # The supplied angel.full.npz calls its training split "train".
        key = 'train_split' if 'train_split' in archive.files else 'train'
        if key not in archive.files:
            raise ValueError(f'Expected train_split or train; found {archive.files}')
        dataset = AngelDataset(archive[key])
    if not len(dataset):
        raise ValueError('No usable training samples after preprocessing')
    print(f'Category={category}, split={key}, usable={len(dataset)}, '
          f'filtered={dataset.rejected}', flush=True)

    weights_dir = PROJECT_ROOT / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f'{category}.ckpt'
    if weights_path.exists():
        print(f'Final checkpoint will replace {weights_path}', flush=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        num_workers=0, pin_memory=True, collate_fn=ThreePointAbs.collate)
    model = AngelModel(repr=ReprType.threeseqabs, modeltype=SketchDiffusion.ModelType.birnn,
        time_embedding=SketchDiffusion.TimeEmbedding.randomfourier,
        arch_layer=3, arch_internal=96, arch_pe_dim=8, arch_dropout=0., noise_T=35,
        optim_lr=1e-3, optim_gamma=.995, optim_interval='epoch', test_save_everything=False)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epoch, precision=32,
        gradient_clip_val=.1, limit_val_batches=0, num_sanity_val_steps=0,
        enable_checkpointing=False, logger=CSVLogger(str(PROJECT_ROOT / 'logs'), name=category),
        callbacks=[EpochProgress()], log_every_n_steps=min(20, len(loader)))
    trainer.fit(model, train_dataloaders=loader)
    # Save the final epoch, rather than selecting a model with held-out data.
    trainer.save_checkpoint(str(weights_path))
    print(f'Saved weights: {weights_path}', flush=True)
    del model
    trained = AngelModel.load_from_checkpoint(str(weights_path), map_location='cpu').cuda()
    now = datetime.now(CHINA_TIME)
    image_dir = PROJECT_ROOT / 'imgs_gen' / category / now.strftime('%Y-%m-%d')
    image_dir.mkdir(parents=True, exist_ok=True)
    run_id = now.strftime('%H%M%S_%f')
    rng = np.random.default_rng(random_seed)
    # Lengths follow the training distribution; coordinates start from noise.
    lengths = rng.choice([len(x) for x in dataset.samples], size=num_images)
    # Bound sampling memory even when a large number of images is requested.
    sample_batch_size = min(batch_size, 32)
    for start in range(0, num_images, sample_batch_size):
        points, lens = trained.generate(lengths[start:start + sample_batch_size],
                                        seed=(random_seed + start) % 2**32)
        if not np.isfinite(points).all():
            raise RuntimeError('Non-finite generated coordinates')
        for offset, (sample, length) in enumerate(zip(points, lens)):
            save_image(sample, length, image_dir / f'{run_id}_{start + offset + 1:06d}.png')
    metadata = dict(npz_path=str(data_path), batch_size=batch_size, random_seed=random_seed,
        epoch=epoch, num_images=num_images, category=category, split=key,
        usable_samples=len(dataset), filtered_samples=dataset.rejected,
        weights=str(weights_path), global_step=trainer.global_step)
    (image_dir / f'{run_id}_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Generated {num_images} images: {image_dir}', flush=True)
    return weights_path, image_dir


if __name__ == '__main__':
    train_and_generate(**CONFIG)
