"""Mask guidance and the six-neighbour MIND constraint from Appendix S1."""

import torch
import torch.nn.functional as F


def mask_guidance(real, synthetic, mask):
    """Mean over ALL patch voxels, as in the two equations in Appendix S1.

    Do not divide by the number of vascular/background voxels: that defines
    a different weighting. Inputs are single-channel tensors in [-1, 1].
    """
    outside = F.l1_loss(real * (1.0 - mask), synthetic * (1.0 - mask))
    inside = F.l1_loss(real * mask, synthetic * mask)
    return outside + (2.0 - inside)


def erode_mask(mask):
    """One-voxel binary erosion with a 3 x 3 x 3 structuring element."""
    kernel = mask.new_ones((1, 1, 3, 3, 3))
    return (F.conv3d(mask, kernel, padding=1) == 27).to(mask.dtype)


def mind_descriptor(image, patch_size=3, eps=1e-6):
    """Six axial-neighbour MIND responses, retaining the source roll boundary.

    Sum pooling computes the same local squared distances as the supplied
    unfold implementation without materialising every 3-D patch.
    """
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError('MIND patch_size must be a positive odd integer.')
    pad = patch_size // 2
    shifts = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
              (0, -1, 0), (0, 0, 1), (0, 0, -1))
    distances = []
    for shift in shifts:
        shifted = torch.roll(image, shifts=shift, dims=(2, 3, 4))
        squared = F.pad((image - shifted).square(), (pad,) * 6, mode='replicate')
        distances.append(F.avg_pool3d(squared, patch_size, stride=1) * patch_size ** 3)
    distances = torch.cat(distances, dim=1)
    variance = distances.mean(dim=1, keepdim=True) + eps
    descriptor = torch.exp(-distances / variance)
    return descriptor / descriptor.amax(dim=1, keepdim=True).clamp_min(1e-8)


def core_mind_loss(synthetic_ncct, cta_mask, min_core_voxels=10):
    """Per-sample (1 - mean MIND) inside the eroded CTA mask.

    Retains the source safeguard: samples with fewer than ten core voxels
    contribute zero. A connected zero keeps empty-mask batches differentiable.
    The stage-1 model applies the manuscript weight of ten.
    """
    core = erode_mask(cta_mask)
    descriptor = mind_descriptor(synthetic_ncct)
    count = core.sum(dim=(1, 2, 3, 4))
    mean = (descriptor * core).sum(dim=(1, 2, 3, 4)) / (6 * count).clamp_min(1)
    return torch.where(count >= min_core_voxels, 1.0 - mean, mean * 0.0).mean()
