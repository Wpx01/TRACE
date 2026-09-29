"""CPU smoke tests: python -m stage1.tests (no patient data or weights needed)."""

import csv
import io
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from .common import learning_rate, load_generator
from .data import PairedPatchDataset, sitk_filename
from .infer import infer_file, patch_starts, predict_array
from .losses import core_mind_loss, mask_guidance, mind_descriptor
from .model import Stage1Model
from .networks import make_generator
from .train import main as train_stage1
from .infer import main as infer_main
from stage2.model import Stage2Model, patch_minmax
from stage2.train import main as train_stage2


@contextmanager
def temporary_working_directory():
    """Allow legacy NIfTI I/O under Unicode temporary parent directories."""
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            yield directory
        finally:
            os.chdir(previous)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(11)

    def test_mask_mean_uses_entire_patch(self):
        real = torch.zeros(1, 1, 4, 4, 4)
        synthetic = torch.ones_like(real)
        mask = torch.zeros_like(real)
        mask[:, :, 0] = 1
        self.assertAlmostEqual(mask_guidance(real, synthetic, mask).item(), 2.5)

    def test_mind_matches_source_unfold_distances_and_gradient(self):
        image = torch.randn(2, 1, 6, 7, 8, requires_grad=True)
        shifts = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                  (0, -1, 0), (0, 0, 1), (0, 0, -1))
        distances = []
        for shift in shifts:
            other = torch.roll(image, shift, (2, 3, 4))
            first = F.pad(image, (1,) * 6, mode='replicate')
            second = F.pad(other, (1,) * 6, mode='replicate')
            for dimension in (2, 3, 4):
                first = first.unfold(dimension, 3, 1)
                second = second.unfold(dimension, 3, 1)
            distances.append(((first - second) ** 2).sum(dim=(-1, -2, -3)))
        distances = torch.cat(distances, dim=1)
        reference = torch.exp(-distances / (distances.mean(dim=1, keepdim=True) + 1e-6))
        reference = reference / reference.amax(dim=1, keepdim=True).clamp_min(1e-8)
        actual = mind_descriptor(image)
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
        gradient = torch.autograd.grad(actual.mean(), image, retain_graph=True)[0]
        expected = torch.autograd.grad(reference.mean(), image)[0]
        torch.testing.assert_close(gradient, expected, atol=2e-7, rtol=1e-4)

    def test_empty_core_is_finite_and_differentiable(self):
        image = torch.randn(1, 1, 6, 6, 6, requires_grad=True)
        loss = core_mind_loss(image, torch.zeros_like(image))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_minmax_is_per_sample(self):
        samples = torch.stack((torch.arange(27), torch.arange(27) * 8 + 100)).float().reshape(2, 1, 3, 3, 3)
        normalized = patch_minmax(samples)
        torch.testing.assert_close(normalized[0], normalized[1])
        self.assertEqual(normalized.min().item(), 0)
        self.assertEqual(normalized.max().item(), 1)
        self.assertEqual(patch_minmax(torch.ones_like(samples)).sum().item(), 0)

    def test_stage1_updates_both_generators_and_discriminators(self):
        model = Stage1Model(ngf=2, ndf=2, pool_size=2)
        before = {name: next(getattr(model, name).parameters()).detach().clone()
                  for name in ('G_A', 'G_B', 'D_A', 'D_B')}
        batch = {'cta': torch.rand(1, 1, 32, 32, 32) * 2 - 1,
                 'ncct': torch.rand(1, 1, 32, 32, 32) * 2 - 1}
        mask = torch.zeros_like(batch['cta'])
        mask[:, :, 8:24, 8:24, 8:24] = 1
        batch.update(cta_mask=mask, ncct_mask=mask)
        losses = model.train_step(batch, model.make_optimizers())
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        for name in before:
            self.assertFalse(torch.equal(before[name], next(getattr(model, name).parameters())))

    def test_stage2_freezes_teacher_parameters_and_running_statistics(self):
        model = Stage2Model(ngf=2)
        model.train()
        teacher = {key: value.clone() for key, value in model.G_A.state_dict().items()}
        student = next(model.G_B.parameters()).detach().clone()
        batch = {'cta': torch.rand(1, 1, 16, 16, 16) * 2 - 1,
                 'ncct': torch.rand(1, 1, 16, 16, 16) * 2 - 1}
        losses = model.train_step(batch, model.make_optimizers())
        self.assertEqual(set(losses), {'silhouette'})
        self.assertTrue(np.isfinite(losses['silhouette']))
        self.assertFalse(model.G_A.training)
        for key, value in model.G_A.state_dict().items():
            self.assertTrue(torch.equal(teacher[key], value), key)
        self.assertTrue(all(parameter.grad is None for parameter in model.G_A.parameters()))
        self.assertFalse(torch.equal(student, next(model.G_B.parameters())))

    def test_checkpoint_loading_and_verified_legacy_aliases(self):
        network = make_generator(2)
        state = network.state_dict()
        state['input_block.1.weight'] = state['model.1.weight'].clone()
        state = {'module.' + key: value for key, value in state.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save(state, path)
            restored = make_generator(2)
            load_generator(restored, path, 'G_B')
            for key, value in network.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            state['module.input_block.1.weight'] += 1
            torch.save(state, path)
            with self.assertRaises(ValueError):
                load_generator(restored, path, 'G_B')

    def test_learning_rate_boundaries(self):
        self.assertEqual(learning_rate(1, 0, 500, 500), 1)
        self.assertEqual(learning_rate(1, 499, 500, 500), 1)
        self.assertAlmostEqual(learning_rate(1, 750, 500, 500), 0.5)
        self.assertEqual(learning_rate(1, 1000, 500, 500), 0)
        self.assertEqual(learning_rate(1, 100, 50, 50), 0)

    def test_terminal_patch_and_batched_identity_reconstruction(self):
        self.assertEqual(patch_starts(51, 32, 16), [0, 16, 19])
        array = np.random.default_rng(3).uniform(20, 240, (35, 34, 33)).astype(np.float32)
        result = predict_array(torch.nn.Identity(), array, 'cpu', (32,) * 3, (16,) * 3, 3)
        np.testing.assert_allclose(result, array, atol=1e-4, rtol=1e-6)

    def test_small_volume_and_empty_volume(self):
        small = np.full((10, 13, 15), 100, dtype=np.float32)
        result = predict_array(torch.nn.Identity(), small, 'cpu', (32,) * 3, (16,) * 3)
        np.testing.assert_allclose(result, small, atol=1e-5)

        class NeverCalled(torch.nn.Module):
            def forward(self, value):
                raise AssertionError('Empty windows should never run inference.')

        empty = np.zeros((35, 34, 33), dtype=np.float32)
        result = predict_array(NeverCalled(), empty, 'cpu', (32,) * 3, (16,) * 3)
        np.testing.assert_array_equal(result, empty)

    def test_exact_sixty_percent_valid_coverage_is_zero(self):
        array = np.full((48, 32, 32), 100, dtype=np.float32)
        with patch('stage1.infer.is_background', side_effect=[False, False, False, True, True]):
            result = predict_array(torch.nn.Identity(), array, 'cpu', (32,) * 3, (4, 32, 32), 2)
        self.assertTrue(np.all(result[16:32] == 0))
        np.testing.assert_allclose(result[12:16], 100, atol=1e-5)

    def test_nifti_geometry_manifest_masks_and_input_protection(self):
        with temporary_working_directory() as directory:
            folder = Path(directory)
            image = sitk.GetImageFromArray(np.full((33, 34, 35), 100, dtype=np.float32))
            image.SetSpacing((0.7, 0.8, 1.25))
            image.SetOrigin((11, -20, 17))
            image.SetDirection((-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0))
            mask = sitk.GetImageFromArray(np.ones((33, 34, 35), dtype=np.float32))
            mask.CopyInformation(image)
            for name in ('ncct', 'cta'):
                sitk.WriteImage(image, sitk_filename(folder / (name + '.nii.gz')))
                sitk.WriteImage(mask, sitk_filename(folder / (name + '_mask.nii.gz')))
            manifest = folder / 'pairs.csv'
            columns = ['ncct', 'cta', 'ncct_mask', 'cta_mask']
            with manifest.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerow([name + '.nii.gz' for name in columns])
            data = PairedPatchDataset(manifest, True, (32,) * 3)
            self.assertEqual(data[0]['ncct'].shape, (1, 32, 32, 32))
            self.assertEqual(data[0]['ncct_mask'].min().item(), 1)
            self.assertEqual(set(PairedPatchDataset(manifest, False, (32,) * 3)[0]), {'ncct', 'cta'})
            target = folder / 'result.nii.gz'
            infer_file(torch.nn.Identity(), folder / 'ncct.nii.gz', target, 'cpu',
                       (32,) * 3, (16,) * 3, 3)
            loaded = sitk.ReadImage(sitk_filename(target))
            self.assertEqual(image.GetSize(), loaded.GetSize())
            for attribute in ('GetSpacing', 'GetOrigin', 'GetDirection'):
                np.testing.assert_allclose(getattr(image, attribute)(), getattr(loaded, attribute)())
            with self.assertRaises(ValueError):
                infer_file(torch.nn.Identity(), target, target, 'cpu')
            with self.assertRaises(FileExistsError):
                infer_file(torch.nn.Identity(), folder / 'ncct.nii.gz', target, 'cpu')
            mask.SetOrigin((1, 2, 3))
            sitk.WriteImage(mask, sitk_filename(folder / 'ncct_mask.nii.gz'))
            with self.assertRaisesRegex(ValueError, 'geometry'):
                data[0]

    def test_training_entry_points_resume_and_final_inference(self):
        with temporary_working_directory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            rng = np.random.default_rng(7)
            for name in ('ncct', 'cta'):
                image = sitk.GetImageFromArray(rng.uniform(20, 240, (32,) * 3).astype(np.float32))
                mask = sitk.GetImageFromArray(np.ones((32,) * 3, dtype=np.float32))
                sitk.WriteImage(image, name + '.nii.gz')
                sitk.WriteImage(mask, name + '_mask.nii.gz')
            columns = ['ncct', 'cta', 'ncct_mask', 'cta_mask']
            with (root / 'train.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerow([name + '.nii.gz' for name in columns])
            shared = ['--manifest', str(root / 'train.csv'), '--device', 'cpu',
                      '--batch-size', '1', '--patch-size', '32', '32', '32',
                      '--ngf', '2', '--constant-epochs', '0', '--decay-epochs', '2',
                      '--save-every', '1']
            for stage, entry, model_class in ((1, train_stage1, Stage1Model),
                                               (2, train_stage2, Stage2Model)):
                arguments = shared + (['--ndf', '2', '--pool-size', '1'] if stage == 1 else [])
                initial = [] if stage == 1 else [
                    '--teacher', str(root / 'stage1/2_net_G_A.pth'),
                    '--student', str(root / 'stage1/2_net_G_B.pth')]
                output = root / ('stage' + str(stage))
                restarted = root / ('resume' + str(stage))
                entry(arguments + initial + ['--output', str(output)])
                original = model_class.train_step
                calls = [0]

                def interrupt(instance, batch, optimizers):
                    calls[0] += 1
                    if calls[0] == 2:
                        raise RuntimeError('intentional interruption')
                    return original(instance, batch, optimizers)

                with patch.object(model_class, 'train_step', interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'intentional interruption'):
                        entry(arguments + initial + ['--output', str(restarted)])
                entry(arguments + ['--output', str(restarted), '--resume', str(restarted / 'latest.pt')])
                baseline = torch.load(output / 'latest.pt', weights_only=True)
                resumed = torch.load(restarted / 'latest.pt', weights_only=True)
                for name, weights in baseline['networks'].items():
                    for key, value in weights.items():
                        self.assertTrue(torch.equal(value, resumed['networks'][name][key]),
                                        '{}:{}'.format(name, key))
            # Normalization buffers as well as teacher parameters stay unchanged.
            before = torch.load(root / 'stage1/2_net_G_A.pth', weights_only=True)
            after = torch.load(root / 'stage2/2_net_G_A.pth', weights_only=True)
            self.assertTrue(all(torch.equal(value, after[key]) for key, value in before.items()))
            infer_main(['--input', 'ncct.nii.gz', '--output', 'synthetic.nii.gz',
                        '--checkpoint', str(root / 'stage2/2_net_G_B.pth'),
                        '--ngf', '2', '--device', 'cpu', '--patch-size', '32', '32', '32'], stage=2)
            generated = sitk.ReadImage('synthetic.nii.gz')
            self.assertEqual(generated.GetSize(), (32, 32, 32))
            self.assertTrue(np.isfinite(sitk.GetArrayFromImage(generated)).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
