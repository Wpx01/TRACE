"""Bidirectional mask-guided CycleGAN used for TRACE stage 1.

For inherited CycleGAN conventions, see ../THIRD_PARTY_NOTICES.md.
"""

import random
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F

from .losses import core_mind_loss, mask_guidance
from .networks import initialize_weights, make_discriminator, make_generator


class ImagePool:
    """Historical fake-image replay buffer used by the supplied CycleGAN."""

    def __init__(self, capacity=50):
        self.capacity = capacity
        self.images = []

    def query(self, batch):
        if not self.capacity:
            return batch.detach()
        result = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                result.append(image)
            elif random.random() > 0.5:
                index = random.randrange(self.capacity)
                result.append(self.images[index].clone())
                self.images[index] = image.clone()
            else:
                result.append(image)
        return torch.cat(result)


class Stage1Model(nn.Module):
    """A denotes CTA; B denotes NCCT in the original naming convention.

    G_A: CTA -> NCCT (future teacher); G_B: NCCT -> CTA (future student).
    D_A judges NCCT, D_B judges CTA. Discriminators output probabilities.
    """

    def __init__(self, ngf=64, ndf=64, pool_size=50):
        super().__init__()
        self.G_A = initialize_weights(make_generator(ngf))
        self.G_B = initialize_weights(make_generator(ngf))
        self.D_A = initialize_weights(make_discriminator(ndf))
        self.D_B = initialize_weights(make_discriminator(ndf))
        self.ncct_pool = ImagePool(pool_size)
        self.cta_pool = ImagePool(pool_size)

    def make_optimizers(self, lr=8e-4):
        generator = torch.optim.Adam(chain(self.G_A.parameters(), self.G_B.parameters()),
                                     lr=lr, betas=(0.5, 0.999), weight_decay=0)
        discriminator = torch.optim.Adam(chain(self.D_A.parameters(), self.D_B.parameters()),
                                         lr=lr, betas=(0.5, 0.999), weight_decay=0)
        return [generator, discriminator]

    @staticmethod
    def gan_loss(probability, real):
        target = torch.ones_like(probability) if real else torch.zeros_like(probability)
        return F.binary_cross_entropy(probability, target)

    def train_step(self, batch, optimizers):
        optimizer_g, optimizer_d = optimizers
        cta, ncct = batch['cta'], batch['ncct']
        for discriminator in (self.D_A, self.D_B):
            discriminator.requires_grad_(False)
        optimizer_g.zero_grad(set_to_none=True)
        synthetic_ncct = self.G_A(cta)
        reconstructed_cta = self.G_B(synthetic_ncct)
        synthetic_cta = self.G_B(ncct)
        reconstructed_ncct = self.G_A(synthetic_cta)
        # Preserve the source call order because InstanceNorm tracks statistics.
        identity_ncct = self.G_A(ncct)
        identity_cta = self.G_B(cta)
        losses = {
            'gan_A': self.gan_loss(self.D_A(synthetic_ncct), True),
            'gan_B': self.gan_loss(self.D_B(synthetic_cta), True),
            'cycle_A': 10 * F.l1_loss(reconstructed_cta, cta),
            'cycle_B': 10 * F.l1_loss(reconstructed_ncct, ncct),
            'identity_A': 5 * F.l1_loss(identity_ncct, ncct),
            'identity_B': 5 * F.l1_loss(identity_cta, cta),
            'mask_A': mask_guidance(cta, synthetic_ncct, batch['cta_mask']),
            'mask_B': mask_guidance(ncct, synthetic_cta, batch['ncct_mask']),
            'mind_A': 10 * core_mind_loss(synthetic_ncct, batch['cta_mask']),
        }
        generator_loss = sum(losses.values())
        if not torch.isfinite(generator_loss):
            raise FloatingPointError('Non-finite stage-1 generator loss.')
        generator_loss.backward()
        optimizer_g.step()

        for discriminator in (self.D_A, self.D_B):
            discriminator.requires_grad_(True)
        optimizer_d.zero_grad(set_to_none=True)
        loss_d_a = 0.5 * (self.gan_loss(self.D_A(ncct), True) + self.gan_loss(
            self.D_A(self.ncct_pool.query(synthetic_ncct)), False))
        loss_d_b = 0.5 * (self.gan_loss(self.D_B(cta), True) + self.gan_loss(
            self.D_B(self.cta_pool.query(synthetic_cta)), False))
        discriminator_loss = loss_d_a + loss_d_b
        if not torch.isfinite(discriminator_loss):
            raise FloatingPointError('Non-finite stage-1 discriminator loss.')
        discriminator_loss.backward()
        optimizer_d.step()
        losses.update(total_G=generator_loss, D_A=loss_d_a, D_B=loss_d_b)
        return {key: value.detach().item() for key, value in losses.items()}
