"""Sliding-window synthesis shared by the two TRACE stages."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from .common import load_generator, resolve_device
from .data import check_intensities, pad_to_patch, read_volume, sitk_filename
from .networks import make_generator


def patch_starts(length, patch_size, stride):
    if length < patch_size or not 1 <= stride <= patch_size:
        raise ValueError('Invalid patch extent or stride.')
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def is_background(patch):
    return float(patch.sum(dtype=np.float64)) < 100 or np.count_nonzero(patch) / patch.size < 0.05


def predict_array(generator, array, device, patch_size=(64, 64, 64),
                  stride=(16, 16, 16), batch_size=1):
    """Input/output: float32 X,Y,Z arrays on the 0--255 scale.

    Valid windows contribute uniform weights. At a voxel, more than 60% of
    covering windows must be valid; otherwise its output is zero.
    """
    if array.ndim != 3 or not np.isfinite(array).all() or any(n == 0 for n in array.shape):
        raise ValueError('Expected a finite, non-empty 3-D array.')
    if len(patch_size) != 3 or any(p < 4 or p % 4 for p in patch_size):
        raise ValueError('Patch sizes must be positive multiples of four.')
    if len(stride) != 3 or batch_size < 1:
        raise ValueError('Expected three stride values and a positive batch size.')
    check_intensities(array, 'inference input')
    original_shape = array.shape
    array = pad_to_patch(array.astype(np.float32, copy=False), patch_size)
    # The source pads small arrays first, then replicates an odd terminal z slice.
    if array.shape[2] % 2:
        array = np.pad(array, ((0, 0), (0, 0), (0, 1)), mode='edge')
    coordinates = [patch_starts(n, p, s) for n, p, s in zip(array.shape, patch_size, stride)]
    prediction_sum = np.zeros_like(array, dtype=np.float32)
    covering = np.zeros_like(array, dtype=np.uint32)
    contributing = np.zeros_like(array, dtype=np.uint32)
    pending_regions, pending_patches = [], []
    generator.eval()

    def flush():
        if not pending_patches:
            return
        # Shape must be N,1,X,Y,Z, including a partial last batch.
        inputs = np.stack(pending_patches, axis=0)[:, None] / 127.5 - 1.0
        tensor = torch.from_numpy(inputs).to(device)
        with torch.inference_mode():
            output = generator(tensor)
        if tuple(output.shape) != tuple(tensor.shape) or not torch.isfinite(output).all():
            raise ValueError('Generator returned an invalid prediction shape or non-finite values.')
        predictions = (output[:, 0].cpu().numpy() * 127.5 + 127.5).clip(0, 255)
        for region, prediction in zip(pending_regions, predictions):
            prediction_sum[region] += prediction
            contributing[region] += 1
        pending_regions.clear()
        pending_patches.clear()

    for start in product(*coordinates):
        region = tuple(slice(s, s + p) for s, p in zip(start, patch_size))
        covering[region] += 1
        patch = array[region]
        if not is_background(patch):
            pending_regions.append(region)
            pending_patches.append(patch)
            if len(pending_patches) == batch_size:
                flush()
    flush()
    # Integer comparison implements the strict > 0.6 threshold exactly.
    keep = (contributing.astype(np.uint64) * 5 > covering.astype(np.uint64) * 3)
    result = np.zeros_like(prediction_sum)
    np.divide(prediction_sum, contributing, out=result, where=keep)
    return result[tuple(slice(0, n) for n in original_shape)]


def infer_file(generator, source, destination, device, patch_size=(64, 64, 64),
               stride=(16, 16, 16), batch_size=1, overwrite=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('Input and output must be different files.')
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    array, image = read_volume(source)
    result = predict_array(generator, array, device, patch_size, stride, batch_size)
    output = sitk.GetImageFromArray(result.transpose(2, 1, 0).copy())
    output.CopyInformation(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, sitk_filename(destination))
    return result.shape


def main(argv=None, stage=1):
    parser = argparse.ArgumentParser(description='TRACE stage-{} volumetric inference'.format(stage),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=True, help='Preprocessed .nii/.nii.gz file or directory')
    parser.add_argument('--output', required=True, help='Output NIfTI file or directory')
    parser.add_argument('--checkpoint', required=True, help='G_B checkpoint (G_A for cta-to-ncct)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--patch-size', type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument('--stride', type=int, nargs=3, default=[16, 16, 16])
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing output volumes')
    if stage == 1:
        parser.add_argument('--direction', choices=['ncct-to-cta', 'cta-to-ncct'], default='ncct-to-cta')
    args = parser.parse_args(argv)
    source, destination = Path(args.input), Path(args.output)
    if source.is_dir():
        sources = sorted(path for path in source.iterdir()
                         if path.is_file() and (path.name.endswith('.nii') or path.name.endswith('.nii.gz')))
        if not sources:
            parser.error('Input directory contains no .nii or .nii.gz volumes.')
        pairs = [(path, destination / path.name) for path in sources]
    elif source.is_file():
        if not (destination.name.endswith('.nii') or destination.name.endswith('.nii.gz')):
            parser.error('A single input file requires a .nii or .nii.gz output filename.')
        pairs = [(source, destination)]
    else:
        parser.error('Input does not exist.')
    for input_path, output_path in pairs:
        if input_path.resolve() == output_path.resolve():
            parser.error('Output would overwrite the input.')
        if output_path.exists() and not args.overwrite:
            parser.error('Output already exists: {}'.format(output_path))
    if args.ngf < 1:
        parser.error('ngf must be positive.')
    device = resolve_device(args.device)
    generator = make_generator(args.ngf)
    name = 'G_A' if getattr(args, 'direction', '') == 'cta-to-ncct' else 'G_B'
    load_generator(generator, args.checkpoint, name)
    generator.to(device).eval()
    for input_path, output_path in pairs:
        shape = infer_file(generator, input_path, output_path, device, tuple(args.patch_size),
                           tuple(args.stride), args.batch_size, args.overwrite)
        print('{} -> {} (XYZ={})'.format(input_path.name, output_path, shape), flush=True)


if __name__ == '__main__':
    main()
