"""Explicitly paired NIfTI inputs on the preprocessed 0--255 intensity scale.

No NCCT--CTA registration or implicit resampling is performed here. Tensor
spatial order is X,Y,Z, matching the original Step1/Step2 implementation.
"""

import csv
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


def sitk_filename(path):
    """Use an ASCII relative path when Windows' NIfTI backend needs one.

    Older SimpleITK NIfTI libraries cannot reliably handle non-ASCII absolute
    paths on Windows. A Unicode parent directory is usable when the current
    working directory allows an ASCII relative path to the file.
    """
    value = str(path)
    if os.name == 'nt' and not value.isascii():
        try:
            relative = os.path.relpath(value)
        except ValueError:
            relative = value
        if relative.isascii():
            return relative
        raise ValueError('This SimpleITK release needs ASCII NIfTI paths on Windows. '
                         'Use ASCII filenames and run from a shared parent directory: ' + value)
    return value


def read_volume(path):
    image = sitk.ReadImage(sitk_filename(path), sitk.sitkFloat32)
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Expected a scalar 3-D NIfTI volume: {}'.format(path))
    array = sitk.GetArrayFromImage(image).transpose(2, 1, 0).copy()
    if not np.isfinite(array).all():
        raise ValueError('Non-finite intensities in {}'.format(path))
    return array, image


def check_intensities(array, path):
    if array.min() < -1e-4 or array.max() > 255.0001:
        raise ValueError('{} is not on the required preprocessed 0--255 scale. '
                         'Raw HU volumes require the study preprocessing first.'.format(path))


def read_mask(path, reference):
    array, image = read_volume(path)
    for attribute in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
        if not np.allclose(getattr(image, attribute)(), getattr(reference, attribute)(),
                           rtol=0, atol=1e-4):
            raise ValueError('Mask geometry differs from its source image: {}'.format(path))
    if not np.all(np.isclose(array, 0) | np.isclose(array, 1)):
        raise ValueError('Masks must contain only binary values 0 and 1: {}'.format(path))
    return (array > 0.5).astype(np.float32)


def pad_to_patch(array, patch_size):
    widths = tuple((0, max(0, p - n)) for n, p in zip(array.shape, patch_size))
    return np.pad(array, widths, mode='constant')


class PairedPatchDataset(Dataset):
    """Pair by explicit manifest row, never by independently sorted filenames.

    One randomly sampled patch per case per epoch by default. Sampling controls
    are release defaults (the historical training entry point was not supplied).
    A bounded retry loop reports unusable data instead of hanging indefinitely.
    """

    def __init__(self, manifest, with_masks, patch_size=(64, 64, 64),
                 patches_per_volume=1, min_vessel_voxels=262,
                 background_keep_probability=0.05, max_crop_attempts=256,
                 seed=0):
        self.manifest = Path(manifest).resolve()
        self.with_masks = with_masks
        self.patch_size = tuple(patch_size)
        if len(self.patch_size) != 3 or any(p < 32 or p % 4 for p in self.patch_size):
            raise ValueError('Patch sizes must be multiples of four and at least 32.')
        if patches_per_volume < 1 or max_crop_attempts < 1 or min_vessel_voxels < 0:
            raise ValueError('Invalid crop/sampling counts.')
        if not 0 <= background_keep_probability <= 1:
            raise ValueError('background_keep_probability must be in [0, 1].')
        self.patches_per_volume = patches_per_volume
        self.min_vessel_voxels = min_vessel_voxels
        self.background_keep_probability = background_keep_probability
        self.max_crop_attempts = max_crop_attempts
        self.seed = seed
        self.epoch = 0
        required = ['ncct', 'cta'] + (['ncct_mask', 'cta_mask'] if with_masks else [])
        self.records = []
        with self.manifest.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if not set(required).issubset(reader.fieldnames or []):
                raise ValueError('Manifest needs columns: ' + ', '.join(required))
            seen = set()
            for number, row in enumerate(reader, start=2):
                record = {}
                for key in required:
                    value = (row.get(key) or '').strip()
                    if not value:
                        raise ValueError('Missing {} at manifest row {}'.format(key, number))
                    path = Path(value)
                    if not path.is_absolute():
                        path = self.manifest.parent / path
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    record[key] = path.resolve()
                pair = (record['ncct'], record['cta'])
                if pair in seen:
                    raise ValueError('Duplicate paired examination at row {}'.format(number))
                seen.add(pair)
                self.records.append(record)
        if not self.records:
            raise ValueError('Manifest contains no examinations.')

    def __len__(self):
        return len(self.records) * self.patches_per_volume

    def __getitem__(self, index):
        record = self.records[index // self.patches_per_volume]
        ncct, ncct_image = read_volume(record['ncct'])
        cta, cta_image = read_volume(record['cta'])
        check_intensities(ncct, record['ncct'])
        check_intensities(cta, record['cta'])
        if ncct.shape != cta.shape:
            raise ValueError('Paired arrays must have matching matrix sizes before cropping. '
                             'This loader does not register or resize them: {}'.format(record))
        arrays = {'ncct': ncct, 'cta': cta}
        if self.with_masks:
            arrays['ncct_mask'] = read_mask(record['ncct_mask'], ncct_image)
            arrays['cta_mask'] = read_mask(record['cta_mask'], cta_image)
        arrays = {key: pad_to_patch(value, self.patch_size) for key, value in arrays.items()}
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        shape = arrays['cta'].shape
        for _ in range(self.max_crop_attempts):
            start = [int(rng.integers(0, n - p + 1)) for n, p in zip(shape, self.patch_size)]
            region = tuple(slice(s, s + p) for s, p in zip(start, self.patch_size))
            cropped = {key: value[region] for key, value in arrays.items()}
            # Source RandomCrop filtered CTA voxels above 10 at the 5% threshold.
            if np.mean(cropped['cta'] > 10) < 0.05:
                continue
            if self.with_masks and cropped['cta_mask'].sum() < self.min_vessel_voxels:
                if rng.random() > self.background_keep_probability:
                    continue
            return {key: torch.from_numpy(np.ascontiguousarray(
                (value / 127.5 - 1.0) if key in ('ncct', 'cta') else value
            )[None]).float() for key, value in cropped.items()}
        raise ValueError('No acceptable patch after {} attempts for {}. Check preprocessing '
                         'and masks, or explicitly adjust sampling controls.'.format(
                             self.max_crop_attempts, record['cta']))
