"""Stage 2: frozen CTA-to-NCCT teacher, silhouette-only student training."""

import torch
from torch import nn
from monai.losses import SSIMLoss

from stage1.networks import make_generator


def patch_minmax(tensor):
    minimum = tensor.amin(dim=(2, 3, 4), keepdim=True)
    maximum = tensor.amax(dim=(2, 3, 4), keepdim=True)
    return (tensor - minimum) / (maximum - minimum + 1e-8)


class Stage2Model(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.G_A = make_generator(ngf)
        self.G_B = make_generator(ngf)
        self.G_A.requires_grad_(False)
        self.G_A.eval()
        self.ssim = SSIMLoss(spatial_dims=3, data_range=1.0, reduction='mean')

    def train(self, mode=True):
        super().train(mode)
        # Freezing parameters alone is insufficient for tracked InstanceNorm.
        self.G_A.eval()
        return self

    def make_optimizers(self, lr=2e-5):
        return [torch.optim.Adam(self.G_B.parameters(), lr=lr,
                                 betas=(0.5, 0.999), weight_decay=0)]

    def silhouette_loss(self, cta, ncct):
        with torch.no_grad():
            synthetic_ncct = self.G_A(cta)
            teacher_silhouette = patch_minmax(cta - synthetic_ncct)
        synthetic_cta = self.G_B(ncct)
        student_silhouette = patch_minmax(synthetic_cta - ncct)
        # Full patch, no vascular mask multiplication and no registration.
        return self.ssim(teacher_silhouette, student_silhouette)

    def train_step(self, batch, optimizers):
        optimizer = optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        loss = self.silhouette_loss(batch['cta'], batch['ncct'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite stage-2 silhouette loss.')
        loss.backward()
        optimizer.step()
        return {'silhouette': loss.detach().item()}
