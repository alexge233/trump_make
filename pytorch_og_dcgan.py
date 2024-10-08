"""
To run this template just do:
python dcgan.py
After a few epochs, launch TensorBoard to see the images being generated at every batch:
tensorboard --logdir default

TODO: REWORK THIS TO WORK!!!
"""
import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

#
# TODO: convert into a jupyter notebook
#       add the fixes from `dcgan` and 
#       cleanup printing on screen
#
def compute_mean_std(dataloader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    count=0
    for images, _ in dataloader:
        count+=1
        batch_samples = images.size(0)  # Get batch size
        for i in range(3):  # Loop over RGB channels
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()

    mean /= count
    std /= count
    return mean, std


class Generator(nn.Module):
    """
    Generator Module for DC GAN.
    This is a model that takes as input a Tensor of random data.
    For brevity, that tensor should be the same dimensions as the image, but that's
    not really a requirement in any way.

    The vanilla DC-GAN design starts by convolving the input to a high channel dimensionality,
    and then keeps convolving to lower channels. It uses a stride of 2, which allows the network
    to essentially learn its own downsampling approach.

    The PyTorch implementation uses `ConvTranspose2d` and the papers also suggest its use.
    The PyTorch-lightning uses `Upsample` which is a non-learnable technique, and probably not the one to use here.

    This is what the design should look like:
        https://pytorch.org/tutorials/_images/dcgan_generator.png

    PyTorch tutorial can be found here:
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    PyTorch-Lightning (is significantly different):
        https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/dcgan.py

    The vanilla was a different beast; see here:
            https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/

    I have adapted it to create 3-channel 128x128 pixel images.
    """
    def __init__(self, nz = 100, ngf = 128, nc = 3):
        """
        Args:
            nz:  number of latent space input (what specifies the random input tensor)
                 there's no clear reason as to why a larger noise input will help
                 Ultimately, we can plug a Transformer if this is to be used for text prompts.
            ngf: numer of generator features (64) used to produce (64 x 64)
            nc:  number of channels in the output (BGR)

        """
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels  = nz,
                               out_channels = ngf * 8,
                               kernel_size  = 4,
                               stride       = 1,
                               padding      = 0,
                               bias         = False),
            nn.BatchNorm2d(ngf * 8, momentum=0.8),
            nn.PReLU(),
            # state size. (ngf * 8 = 512) x 4 x 4
            nn.ConvTranspose2d(in_channels  = ngf * 8,
                               out_channels = ngf * 4,
                               kernel_size  = 4,
                               stride       = 2,
                               padding      = 1,
                               bias         = False),
            nn.BatchNorm2d(ngf * 4, momentum=0.8),
            nn.PReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels  = ngf * 4,
                               out_channels = ngf * 2,
                               kernel_size  = 4,
                               stride       = 2,
                               padding      = 1,
                               bias         = False),
            nn.BatchNorm2d(ngf * 2, momentum=0.8),
            nn.PReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels  = ngf * 2,
                               out_channels = ngf,
                               kernel_size  = 4,
                               stride       = 2,
                               padding      = 1,
                               bias         = False),
            nn.BatchNorm2d(ngf, momentum=0.8),
            nn.PReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels = ngf,
                               out_channels = nc,
                               kernel_size  = 4,
                               stride       = 2,
                               padding      = 1,
                               bias         = False),
            nn.Tanh()
            # state size. (nc) x 64 x 64 (-> 3, 64, 64)
        )
        """
        Output will be:
            Batch, Channels, Height, Width
        """


    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """
    Discriminator tells us if the Generator output is real or fake.
    The Discriminator takes as input the Generator's output (a 3 x 64 x64 Tensor) representing
    a fake image.

    Architecture is similar to a CNN (actually, it is a CNN!):
      -> Conv2D, BatchNorm2D, LeakyReLU (repeats 5 times).
      -> Note; DO NOT use BatchNorm2D on the input layer
      -> Uses Sigmoid instead of Tanh, to Score the input plausibility (0 fake, 1 real)

    """
    def __init__(self, nc = 3, ndf = 64):
        super(Discriminator, self).__init__()

        """
        I believe that the Vanilla DC-GAN used BatchNorm but no dropout.
        This one here adds BatchNorm programmatically, but hass explicit dropout as well.

        The Discriminator returns a `validity` which basically defines if the Generator
        Output is `valid` or `fake`. The higher output implies that the Discriminator
        thinks all Generator outputs are valid.
        >>> TODO: FIX Architecture
        """
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels  = nc,
                      out_channels = ndf,
                      kernel_size  = 4,
                      stride       = 2,
                      padding      = 1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels = ndf,
                      out_channels = ndf * 2,
                      kernel_size  = 4,
                      stride       = 2,
                      padding      = 1,
                      bias         = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels  = ndf * 2,
                      out_channels = ndf * 4,
                      kernel_size  = 4,
                      stride       = 2,
                      padding      = 1,
                      bias         = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels  = ndf * 4,
                      out_channels = ndf * 8,
                      kernel_size  = 4,
                      stride       = 2,
                      padding      = 1,
                      bias         = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # state size. (ndf * 8) x 4 x 4
            nn.Conv2d(in_channels  = ndf * 8,
                      out_channels = 1,
                      kernel_size  = 3, #4
                      stride       = 1,
                      padding      = 0,
                      bias         = False),
            nn.Flatten(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )


    def forward(self, img):
        """
        Returns:
            Tensor (Batch, 1) where each entry in batch has a Binary value
        """
        return self.main(img)


class DCGAN(LightningModule):
    """
    Put Generator and Discriminator in a single class,
    then wrap the training logic and loss methods here.
    Courtesy of Pytorch-Lightining:
    https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/dcgan.py
    """
    def __init__(self,
                 lr: float,
                 batch_size: int,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.batch_size = batch_size
        img_shape = (3, 64, 64)
        self.generator = Generator()
        self.discriminator = Discriminator()


    def forward(self, z):
        return self.generator(z)


    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        """
        >>> PyTorch lightning has a different approach which I'm not sure is correct.
            It averages real loss and fake loss.
            Original DC-GAN simply sums the losses.

        Check here for the main loop to see why the metrics are off:
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

        >>> Both Generated Images and Real images are of shape:
            [64, 3, 64, 64] meaning; Batch, NC, H, W

        >>> However, `valid` is [64, 1] and `fake` is [64, 1]
        """
        criterion = nn.BCELoss()

        z = torch.randn(imgs.shape[0], 100, 1, 1)
        z = z.type_as(imgs)
        #
        # generate fake images from noise as input
        #
        output   = self.generator(z)
        self.fakes = output

        #
        # ground truth result (ie: all fake) or all valid
        # put on GPU because we created this tensor inside training_loop
        #
        valid = torch.full((imgs.size(0),), 1, dtype=torch.float, device=self.device)
        #
        # how well can it label as fake?
        #
        fake = torch.full((imgs.shape[0],), 0, dtype=torch.float, device=self.device)

        # 
        # Update Generator
        #
        if optimizer_idx == 0:

            #
            # Generator Loss is BCE over Generated Fake Images.
            # However, we calculate BCE over the Discriminator's Output over the Fake Images!
            # But we use a `valid` label for it
            #
            fake_out = self.discriminator(output).view(-1)
            g_loss = criterion(fake_out, valid)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'D_G_z2' : fake_out.mean().item(),
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            self.log("generator", g_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return output

        # 
        # Update Discriminator
        #
        if optimizer_idx == 1:
            # discriminator run on fake images and then real images
            fake_out = self.discriminator(output).view(-1)
            real_out = self.discriminator(imgs).view(-1)

            #
            # Real Loss for Discriminator is Discriminator over real images
            #
            real_loss = criterion(real_out, valid)

            #
            # Fake Loss for Discriminator is Discriminator over fake images
            #
            fake_loss = criterion(fake_out, fake)

            #
            # discriminator loss is the SUM (not the average)
            #
            d_loss = (real_loss + fake_loss)

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'D_G_z1' : fake_out.mean().item(),
                'D_x' : real_out.mean().item(),
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            self.log("discriminator", d_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return output


    def configure_optimizers(self):
        """
        Setup Learning rates, etc
        """
        lr = self.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []


    def train_dataloader(self):
        """
        Setup Dataloader (pytorch-lightining feature).
        >>> TODO: hack into this and either use Trump Dataset or Alien Circles Dataset!
            This can be done by adding an argumnet to the class init
        """
        dataset = dset.ImageFolder(root="data/trump",
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.6337, 0.4399, 0.3807),
                                   (0.2056, 0.1687, 0.1635)
                               ),
                           ]))
        return DataLoader(dataset,
                          batch_size = self.batch_size,
                          num_workers = 16,
                          shuffle = True,
                          persistent_workers = True)


    def train_epoch_end(self):
        """
        On Epoch End, sample images and push them to Tensorboard in a grid for visualisation
        """
        # log sampled images (only six from the looks of it)
        grid = torchvision.utils.make_grid(self.fakes, nrow=32, normalize=False)
        #
        # default dataformats is `(N,3,H,W)` which in this case is
        # Batch (or grid samples), C, H (64) Width (the Sample image grid concatenated)
        #
        self.logger.experiment.add_image(tag = 'generated_images',
                                         img_tensor = grid,
                                         global_step = 0)


def main(args: Namespace) -> None:
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    #model = DCGAN(**vars(args))
    logger = TensorBoardLogger("default", name="DCGAN")
    model = DCGAN(lr=0.0001, batch_size=32)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(accelerator="gpu", devices=1, logger=logger, max_epochs=1000, precision=32)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="dimensionality of the latent space")

    hparams = parser.parse_args()
    main(hparams)
