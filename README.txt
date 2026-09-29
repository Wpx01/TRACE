TRACE - Quick Start

Run all commands from the project folder containing requirements.txt.
Keep stage1 and stage2 together. Data and pretrained weights are not included.


1. Data preparation

Prepare paired NCCT and CTA volumes in .nii or .nii.gz format. Each pair must have the same image dimensions. 
Apply the preprocessing to both NCCT and CTA before running the code:

  a. Resample the images to 0.5 x 0.5 x 0.5 mm voxels using B-spline interpolation.
  b. Apply a window level of 130 HU and a window width of 800 HU:
     clip values below -270 HU to -270 and values above 530 HU to 530.
  c. Linearly map this fixed HU range to 0-255:

       image_255 = (clip(image_HU, -270, 530) + 270) * 255 / 800

Save these preprocessed volumes as the inputs listed in train.csv.

For stage1, also prepare a binary vessel mask (0 or 1) for each volume.
Each mask must match its own image's dimensions, spacing, origin and direction.
Stage2 only needs the paired NCCT and CTA volumes.

Example folder structure:

  data/
    ncct/case001.nii.gz
    cta/case001.nii.gz
    ncct_masks/case001.nii.gz
    cta_masks/case001.nii.gz

Create train.csv in the project folder:

ncct,cta,ncct_mask,cta_mask
data/ncct/case001.nii.gz,data/cta/case001.nii.gz,data/ncct_masks/case001.nii.gz,data/cta_masks/case001.nii.gz

Add one row per paired examination and replace the example paths with your own.
Paths are relative to train.csv. Use the same CSV for both training stages;
stage2 ignores the mask columns. Keep training and test patients separate.
On Windows, use English-only folder and file names for NIfTI input/output.


2. Environment

Use 64-bit Python 3.8.10 and an NVIDIA GPU for training.
With Conda installed, create and activate an environment:

conda create -n trace python=3.8.10 pip -y
conda activate trace

Install PyTorch and the remaining dependencies.
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

3. Training

Stage1

python -m stage1.train --manifest train.csv --output checkpoints/stage1 --device cuda:0

The default run trains for 1000 epochs. The two final generator files are:

  checkpoints/stage1/1000_net_G_A.pth
  checkpoints/stage1/1000_net_G_B.pth

Stage2

After stage1 finishes, run:

python -m stage2.train --manifest train.csv --output checkpoints/stage2 --teacher checkpoints/stage1/1000_net_G_A.pth --student checkpoints/stage1/1000_net_G_B.pth --device cuda:0

The default run trains for 100 epochs. Use this file for final inference:

  checkpoints/stage2/100_net_G_B.pth

Training settings are supplied by default. Use a new output folder for each run.


4. Inference

Prepare an NCCT volume on the same 0-255 intensity scale used for training.
No CTA volume or vessel mask is needed.

python -m stage2.infer --input data/ncct/case001.nii.gz --output outputs/case001.nii.gz --checkpoint checkpoints/stage2/100_net_G_B.pth --device cuda:0

To process a folder of NCCT volumes:

python -m stage2.infer --input data/ncct --output outputs --checkpoint checkpoints/stage2/100_net_G_B.pth --device cuda:0

Output files preserve the input image dimensions, spacing, origin and direction.
Output intensities are on the 0-255 scale; they are not converted back to HU.
Use separate output locations when trying the two examples above.
