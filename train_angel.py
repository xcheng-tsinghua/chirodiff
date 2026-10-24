"""Train ChiroDiff on a Sketch-RNN NPZ, preserving its official splits."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from main import SketchDiffusion
from data.dm import ReprType
from data.batching import ThreePointAbs


class AngelDataset(Dataset):
    def __init__(self, raw, limit=None):
        self.samples = []
        self.rejected = 0
        for sample in raw[:limit]:
            sample = np.asarray(sample, dtype=np.float32)
            if sample.ndim != 2 or sample.shape[1] != 3 or not 5 <= len(sample) <= 300 or not np.isfinite(sample).all():
                self.rejected += 1
                continue
            # Sketch-RNN stores relative coordinates and end-of-stroke flags.
            xy = sample[:, :2].cumsum(axis=0)
            xy -= xy[:1].copy()
            radius = np.linalg.norm(xy, axis=1).max()
            if radius <= 0:
                self.rejected += 1
                continue
            xy *= 10.0 / radius
            pen = sample[:, 2:3].copy()
            pen[-1] = 1
            ts = np.linspace(0, 1, len(xy), dtype=np.float32)[:, None]
            # Match ThreePointAbs: omit the initial (origin) point.
            self.samples.append(torch.from_numpy(np.concatenate((xy, pen, ts), axis=1)[1:]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def draw_grid(points, lengths, path):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, sample, length in zip(axes.flat, points, lengths):
        sample = np.asarray(sample)[:int(length)]
        cuts = np.flatnonzero(sample[:, 2] > .8) + 1
        for stroke in np.split(sample[:, :2], cuts):
            if len(stroke) > 1:
                ax.plot(stroke[:, 0], stroke[:, 1], 'k-', lw=1.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class AngelModel(SketchDiffusion):
    def on_fit_start(self):
        self.on_test_start()

    def on_test_start(self):
        from utils import random_fourier_encoding_dyn
        ts = torch.arange(1, self.hp.noise_T + 1, device=self.device, dtype=self.dtype) / self.hp.noise_T
        self.pe = random_fourier_encoding_dyn(ts[None], self.pe_proj_W, scale=4.)
        self.ema.to(self.device)

    def validation_step(self, batch, batch_idx):
        _, (points, lens) = batch
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(1234 + batch_idx)
            data = self.create_batch_with_utilities(points, lens)
        score = self(data['noisy_points'], None, lens, data['noise_t'], None)
        loss = self.ncsn_loss(score, data['target'], lens, data['t'])
        self.log('valid/loss', loss, on_step=False, on_epoch=True, batch_size=len(lens))
        return loss

    @torch.no_grad()
    def generate(self, lengths, seed=2026):
        was_training = self.training
        self.eval()
        self.on_test_start()
        lens = torch.tensor(lengths, device=self.device)
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(seed)
            noise = torch.randn(len(lens), int(lens.max()), 3, device=self.device)
            result = self.reverse_diffusion(noise, None, lens, None,
                self.SamplingAlgo.ddpm, langevin_strength=.75, draw=False)
        self.train(was_training)
        return result['last'].cpu().numpy(), result['lens'].cpu().numpy()


class Preview(pl.Callback):
    def __init__(self, output, lengths, every):
        self.output, self.lengths, self.every = output, lengths, every
        self.started = time.time()

    def on_validation_end(self, trainer, model):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        status = {'epoch': epoch, 'step': trainer.global_step,
                  'elapsed_seconds': round(time.time() - self.started),
                  'metrics': {k: float(v) for k, v in trainer.callback_metrics.items()}}
        (self.output / 'status.json').write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
        if epoch == 1 or epoch % self.every == 0:
            points, lengths = model.generate(self.lengths)
            draw_grid(points, lengths, self.output / f'samples_epoch_{epoch:04d}.png')
            np.savez(self.output / f'samples_epoch_{epoch:04d}.npz', points=points, lengths=lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=r'D:\document\DataSet\quickdraw\raw\angel.full.npz')
    parser.add_argument('--output', default='runs/angel')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=512, help='Samples per split; 0 uses all samples')
    parser.add_argument('--resume')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--preview-every', type=int, default=5)
    args = parser.parse_args()
    if args.sample and not args.resume:
        parser.error('--sample requires --resume CHECKPOINT')
    if args.limit < 0 or args.batch_size < 1 or args.epochs < 1 or args.preview_every < 1:
        parser.error('epochs, batch size and preview interval must be positive; limit must be >= 0')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(5555, workers=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=True, encoding='latin1') as archive:
        train = AngelDataset(archive['train'], args.limit or None)
        valid = AngelDataset(archive['valid'], args.limit or None)
    if len(train) < 16 or not len(valid):
        raise ValueError('Need at least 16 usable training samples and one validation sample')
    print(f'train={len(train)} rejected={train.rejected}, valid={len(valid)} rejected={valid.rejected}', flush=True)
    lengths = [len(train[i]) for i in np.random.default_rng(2026).choice(len(train), 16, replace=False)]
    draw_grid([x.numpy() for x in train.samples[:16]], [len(x) for x in train.samples[:16]], output / 'training_examples.png')
    if args.sample:
        model = AngelModel.load_from_checkpoint(args.resume, map_location='cpu').cuda()
        points, lens = model.generate(lengths)
        draw_grid(points, lens, output / 'generated.png')
        np.savez(output / 'generated.npz', points=points, lengths=lens)
        return
    model = AngelModel(repr=ReprType.threeseqabs, modeltype=SketchDiffusion.ModelType.birnn,
        time_embedding=SketchDiffusion.TimeEmbedding.randomfourier,
        arch_layer=3, arch_internal=96, arch_pe_dim=8, arch_dropout=0.,
        noise_T=35, optim_lr=1e-3, optim_gamma=.995, optim_interval='epoch',
        test_save_everything=False)
    checkpoint = ModelCheckpoint(dirpath=str(output / 'checkpoints'), monitor='valid/loss',
        filename='best-{epoch:03d}', save_last=True, save_top_k=1)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs,
        precision=32, gradient_clip_val=.1, num_sanity_val_steps=0,
        logger=CSVLogger(str(output), name='metrics'),
        callbacks=[checkpoint, LearningRateMonitor(), Preview(output, lengths, args.preview_every)],
        enable_progress_bar=False, log_every_n_steps=20)
    (output / 'run_config.json').write_text(json.dumps(vars(args), indent=2))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=True, collate_fn=ThreePointAbs.collate)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, num_workers=0,
        pin_memory=True, collate_fn=ThreePointAbs.collate)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.resume)
    best = AngelModel.load_from_checkpoint(checkpoint.best_model_path, map_location='cpu').cuda()
    points, lens = best.generate(lengths)
    draw_grid(points, lens, output / 'generated_best.png')
    print(f'Completed. Best checkpoint: {checkpoint.best_model_path}', flush=True)


if __name__ == '__main__':
    main()
