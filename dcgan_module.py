import os
import torch
from pl_bolts.models.gans import DCGAN
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.transforms import v2
from lightning.pytorch.callbacks import RichProgressBar

import argparse

parser = argparse.ArgumentParser(
                    prog='DCGAN Training',
                    description='Trains a DCGAN to generate Fake images. By default those images will be 64x64 RGB.',
                    epilog='Text at the bottom of help')
parser.add_argument('--rootdir', default='data/trump/', type=str, help='directory with images to use for training.', required=True)
parser.add_argument('-e', '--epochs', default=500, type=int, required=False, help='Epochs to train for')
args = parser.parse_args()

if __name__ == "__main__":
    dataroot = args.rootdir

    transform = v2.Compose([
        v2.Resize(64),
        v2.CenterCrop(64),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor(),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        v2.GaussianNoise(),
    ])

    checkpoint_callback = ModelCheckpoint(
            dirpath=".models/",
            save_top_k=1,
            monitor="loss/gen_epoch"
    )
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=12)
    train_dataloader = DataLoader(dataset,
        batch_size = 64,
        num_workers = 16,
        shuffle = True,
        persistent_workers = True
    )
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name="lightning_logs")

    m = DCGAN(
        image_channels=3
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(
        m,
        train_dataloader,
    )
    print(checkpoint_callback.best_model_path)
    img_list = []
    noise = torch.rand(64, 100)
    fake = m(noise)
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    plt.imshow(np.transpose(img_list[-1]))
    plt.title("Fake Images")
    plt.show()
