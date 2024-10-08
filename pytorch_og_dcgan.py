"""
Taken from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
Adjusted with custom networks to increase complexity.
"""

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "data/"

# Number of workers for dataloader
workers = 12
# Batch size during training
batch_size = 64
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 500
# Learning rate for optimizers (HUGE IMPACT)
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.2
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

"""
    Note: papers say train with batch 32 for 5000 epochs!!!
          also ensure that noise is propagated in both G and D!
          Last, try SGD instead of ADAM
"""

#
#normalize = transforms.Normalize(mean=[0.6308107582835936, 0.4385014286334169, 0.38007994109731374],
#                                  std=[0.18569097698754572, 0.15495167947398258, 0.14816380123900705])

#
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=workers)

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

# Compute the mean and std
mean, std = compute_mean_std(dataloader)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())  # Use computed mean and std
])

# Load the dataset with normalization
dataset = dset.ImageFolder(root=dataroot, 
                           transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size, 
                                         shuffle=True, 
                                         num_workers=workers)


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """Generator of Fake Images.
    It uses Pytorch's ConvTranspose2d, which is a *deconvolution*
    also known as a upsampler. 
    The original work defines `nz` as a parameter, that is the latent space size.
    Using `nz` has a great impact; it defines **how much** information is encoded
    in the latent space, before the Deconvolution takes place.
    The second parameter is `ngf` which defines how many features are used to 
    encode that information.

    PyTorch documentation states that `nz` is used as `input_channels` and that the second
    argument (`ngf`) specifies the output channels produced by the deconvolution:
        https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    And documentation for `BatchNorm2d` is at:
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

    I think one thing I've missed is that the Generator outputs 64x64 whereas the Discriminator
    takes in 32x32
    """
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        """
            Note, arguments for `ConvTranspose2d` are:
            Args:
                input_channels
                output_channels
                kernel_size
                stride
                padding
                bias
                output_padding
                bias
                dilation

            Note, arguments for `BatchNorm2d` are:
            Args:
                num_features:
                eps:
                momentum:
                affine:
                track_running_stats:
            """
        
        self.main = nn.Sequential(

            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8, momentum=0.8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=0.8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=0.8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, momentum=0.8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=0.01)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=0.01)

# Try with SGD instead
#optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.9)
#optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=0.9)


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
D_xes    = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):

    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()

        print(len(data))
        print(type(data))

        real_cpu    = data[0].to(device)
        print(type(real_cpu))
        print("input size", real_cpu.size())
        b_size      = real_cpu.size(0)
        print("batch", b_size)
        label       = torch.full((b_size,), real_label, device=device)
        print("label size", label.size())
        print("label data", label)

        # Forward pass real batch through D
        output      = netD(real_cpu) #.view(-1)
        print("output size", output.size())
        print("output data", output)

        output      = output.view(-1)
        print("output size", output.size())
        print("output data", output)

        # Calculate loss on all-real batch
        errD_real   = criterion(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        """
        There is an important question here, why add noise to the batch?
        How does that aid the Generator?
        Obviously what we see is that we pass a random tensor to the Generator
        and then expect it to produce a *real* image.

        I think I need to explore and understand this better, in order to
        be able to optimise the architecture and achieve better performance
        using the Generator/Discriminator approach.
        """

        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)

        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)

        # Calculate G's loss based on this output
        errG = criterion(output, label)

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        D_xes.append(D_x)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
#
#
#
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.plot(D_x,label="D(x)")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

