"""Shared command-line, checkpoint and training utilities for both stages."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairedPatchDataset


def training_parser(stage):
    parser = argparse.ArgumentParser(description='Train TRACE stage {}'.format(stage),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--manifest', required=True, help='CSV of paired preprocessed volumes')
    parser.add_argument('--output', required=True, help='Checkpoint and log directory')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--batch-size', type=int, default=6 if stage == 1 else 8)
    parser.add_argument('--patch-size', type=int, nargs=3, default=[64, 64, 64], metavar=('X', 'Y', 'Z'))
    parser.add_argument('--lr', type=float, default=8e-4 if stage == 1 else 2e-5)
    parser.add_argument('--constant-epochs', type=int, default=500 if stage == 1 else 50)
    parser.add_argument('--decay-epochs', type=int, default=500 if stage == 1 else 50)
    parser.add_argument('--save-every', type=int, default=50 if stage == 1 else 10)
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers (0 is portable)')
    parser.add_argument('--seed', type=int, default=0, help='Release seed; original experiment seed not supplied')
    parser.add_argument('--ngf', type=int, default=64, help='Generator width; use 64 for manuscript model')
    parser.add_argument('--patches-per-volume', type=int, default=1)
    parser.add_argument('--max-crop-attempts', type=int, default=256)
    parser.add_argument('--resume', help='Full latest.pt training checkpoint from this release')
    parser.add_argument('--max-steps-per-epoch', type=int, default=0,
                        help='Diagnostic run limit; 0 uses all batches')
    if stage == 1:
        parser.add_argument('--ndf', type=int, default=64)
        parser.add_argument('--pool-size', type=int, default=50)
        parser.add_argument('--min-vessel-voxels', type=int, default=262)
        parser.add_argument('--background-keep-probability', type=float, default=0.05)
    else:
        parser.add_argument('--teacher', help='Stage-1 epoch-1000 G_A weights')
        parser.add_argument('--student', help='Stage-1 epoch-1000 G_B weights')
    return parser


def resolve_device(value):
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; install the appropriate PyTorch build, '
                           'or pass --device cpu for a small diagnostic run.')
    return device


def seed_all(seed):
    if seed < 0:
        raise ValueError('seed must be non-negative.')
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_generator(network, path, name):
    """Load original *_net_G_A/B.pth or this release's full checkpoint."""
    payload = torch.load(str(path), map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and 'networks' in payload:
        payload = payload['networks'][name]
    elif isinstance(payload, dict) and 'state_dict' in payload:
        payload = payload['state_dict']
    if not isinstance(payload, dict):
        raise ValueError('Expected a generator state_dict: {}'.format(path))
    state = {key[7:] if key.startswith('module.') else key: value for key, value in payload.items()}
    # The experimental Step2 generator registered the SAME layers twice, under
    # model.* and attention-helper aliases. Remove only verified duplicate keys.
    for key in list(state):
        fields = key.split('.')
        canonical = None
        if fields[0] == 'input_block':
            canonical = 'model.' + '.'.join(fields[1:])
        elif fields[0] == 'down_blocks':
            canonical = 'model.{}.'.format(4 + int(fields[1]) * 3 + int(fields[2])) + '.'.join(fields[3:])
        elif fields[0] == 'res_blocks':
            canonical = 'model.{}.'.format(10 + int(fields[1])) + '.'.join(fields[2:])
        elif fields[0] == 'up_blocks':
            canonical = 'model.{}.'.format(19 + int(fields[1]) * 3 + int(fields[2])) + '.'.join(fields[3:])
        elif fields[0] == 'output_block':
            canonical = 'model.{}.'.format(len(network.model) - 3 + int(fields[1])) + '.'.join(fields[2:])
        if canonical is not None:
            if canonical not in state or not torch.equal(state[key], state[canonical]):
                raise ValueError('Inconsistent legacy generator alias: {}'.format(key))
            del state[key]
    network.load_state_dict(state, strict=True)


def learning_rate(base, completed_epochs, constant_epochs, decay_epochs):
    """Rate for the NEXT epoch; reaches zero after the last completed epoch."""
    return base * max(0.0, 1.0 - max(0, completed_epochs - constant_epochs) / decay_epochs)


def capture_rng():
    state = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def run_training(model, args, stage):
    if args.batch_size < 1 or args.workers < 0 or args.save_every < 1:
        raise ValueError('Invalid batch size, worker count or checkpoint interval.')
    if args.lr <= 0 or args.constant_epochs < 0 or args.decay_epochs <= 0:
        raise ValueError('Invalid learning-rate schedule.')
    if args.max_steps_per_epoch < 0:
        raise ValueError('max-steps-per-epoch must be non-negative.')
    device = resolve_device(args.device)
    model.to(device)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Output directory is not empty. Use a new directory or --resume.')
    dataset = PairedPatchDataset(
        args.manifest, with_masks=stage == 1, patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume,
        min_vessel_voxels=getattr(args, 'min_vessel_voxels', 262),
        background_keep_probability=getattr(args, 'background_keep_probability', 0.05),
        max_crop_attempts=args.max_crop_attempts, seed=args.seed)
    optimizers = model.make_optimizers(args.lr)
    names = ('G_A', 'G_B', 'D_A', 'D_B') if stage == 1 else ('G_A', 'G_B')
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        if state.get('stage') != stage:
            raise ValueError('Resume checkpoint belongs to the other stage.')
        # Prevent a seemingly resumed run from silently switching its protocol.
        mutable = {'resume', 'output', 'device', 'workers', 'save_every', 'teacher', 'student'}
        for key, value in vars(args).items():
            if key not in mutable and state['config'].get(key) != value:
                raise ValueError('Resume configuration differs for {}.'.format(key))
        for name in names:
            getattr(model, name).load_state_dict(state['networks'][name], strict=True)
        for optimizer, saved in zip(optimizers, state['optimizers']):
            optimizer.load_state_dict(saved)
        if stage == 1:
            model.ncct_pool.images = [image.to(device) for image in state['ncct_pool']]
            model.cta_pool.images = [image.to(device) for image in state['cta_pool']]
        restore_rng(state['rng'])
        start = state['epoch']
    total_epochs = args.constant_epochs + args.decay_epochs
    if start >= total_epochs:
        raise ValueError('This training checkpoint has already completed the requested schedule.')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), stage=stage)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for epoch in range(start, total_epochs):
        dataset.epoch = epoch
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(args.seed + epoch),
                            drop_last=False)
        rate = learning_rate(args.lr, epoch, args.constant_epochs, args.decay_epochs)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['lr'] = rate
        model.train()
        totals, samples = {}, 0
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            losses = model.train_step(batch, optimizers)
            count = batch['ncct'].shape[0]
            samples += count
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value * count
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
        metrics = dict(epoch=epoch + 1, lr=rate, samples=samples,
                       **{key: value / samples for key, value in totals.items()})
        print(json.dumps(metrics), flush=True)
        with (output / 'training.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(metrics) + '\n')
        state = dict(stage=stage, epoch=epoch + 1, config=vars(args),
                     networks={name: getattr(model, name).state_dict() for name in names},
                     optimizers=[optimizer.state_dict() for optimizer in optimizers], rng=capture_rng())
        if stage == 1:
            state.update(ncct_pool=model.ncct_pool.images, cta_pool=model.cta_pool.images)
        temporary = output / 'latest.pt.tmp'
        torch.save(state, temporary)
        temporary.replace(output / 'latest.pt')
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == total_epochs:
            for name in names:
                torch.save({key: value.detach().cpu() for key, value in
                            getattr(model, name).state_dict().items()},
                           output / '{}_net_{}.pth'.format(epoch + 1, name))
