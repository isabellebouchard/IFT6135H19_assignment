# Inspired from these repos:
#- https://github.com/soumith/ganhacks
#- https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py
#- plotting library from here: https://github.com/igul222/improved_wgan_training/tree/master/tflib
#- https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py

import time
import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch import autograd
import torch
import torchvision
import numpy as np
import tflib as lib
import tflib.save_images
import tflib.plot
from torchvision.utils import save_image
from VAE_q3 import ConvDecoder

BATCH_SIZE = 128
LAMBDA = 10 # Gradient penalty lambda
ITERS = 200000 # number of generator iterations
CRITIC_ITERS = 5 # number of critic iterations per generator iteration

class Generator(nn.Module):
    def __init__(self, output_shape, latent_space_size=100):
        super(Generator, self).__init__()

        self.latent_space_size = latent_space_size
        self.output_shape = output_shape

        # GAN generator and VAE Decoder share the same architecture in this assignment
        self.generator = ConvDecoder(output_shape)

    def forward(self, input):
        return self.generator(input)


class Discriminator(nn.Module):
    def __init__(self, input_shape, latent_space_size=100):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        self.latent_space_size = latent_space_size

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64, 256, kernel_size=5)
        self.fc61 = nn.Linear(256, latent_space_size)
        self.linear = nn.Linear(latent_space_size, 1)

    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x))
        h2 = self.pool2(h1)
        h3 = F.leaky_relu(self.conv3(h2))
        h4 = self.pool4(h3)
        h5 = F.leaky_relu(self.conv5(h4))
        h6 = h5.view(h5.size(0), -1)
        h7 = self.fc61(h6)
        return self.linear(h7)

class GAN(nn.Module):
    def __init__(self, generator, critic):
        super(GAN, self).__init__()

        self.generator = generator
        self.critic = critic

        assert generator.latent_space_size == critic.latent_space_size
    def decode(self, z):
        return self.generator(z)

# For generating samples
def generate_image(frame, netG):
    fixed_noise_128 = torch.randn(128, netG.latent_space_size)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    noisev = fixed_noise_128
    samples = netG(noisev)
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5) #use with tanh
    save_image(samples, 'results/gan/samples/sample_' + str(frame) + '.png')

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def inf_train_gen(train_iter):
    while True:
        try:
            data = next(train_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            train_iter = iter(train_gen)
            data = next(train_iter)
        yield data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_gen = torch.utils.data.DataLoader(
        datasets.SVHN('./data', split='train', download=True, transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)
    global train_iter
    train_iter = iter(train_gen)
    dev_gen = torch.utils.data.DataLoader(
        datasets.SVHN('./data', split='test', download=True, transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,**kwargs)


    netG = Generator(output_shape=(3, 32, 32))
    netD = Discriminator(input_shape=(3, 32, 32))
    gan_model = GAN(netG, netD)
    print(netG)
    print(netD)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpu = 0
    if use_cuda:
        netD = netD.cuda(gpu)
        netG = netG.cuda(gpu)
        gan_model = gan_model.cuda(gpu)

    one = torch.tensor(1).float()
    mone = one * -1
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # Dataset iterator
    gen = inf_train_gen(train_iter)
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    for iteration in range(ITERS):
        start_time = time.time()

        # Update Discriminator network
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # set to False below in netG update
        for i in range(CRITIC_ITERS):
            _data = gen.__next__()
            _data = _data[0]
            netD.zero_grad()

            # train with real data
            _data = _data.reshape(BATCH_SIZE, 3, 32, 32)
            real_data = torch.stack([preprocess(item) for item in _data])

            if use_cuda:
                real_data = real_data.cuda(gpu)
            real_data_v = real_data

            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake data
            noise = torch.randn(BATCH_SIZE, netD.latent_space_size)
            if use_cuda:
                noise = noise.cuda(gpu)

            noisev = noise
            fake = netG(noisev).data
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        # Update Generator network
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        noise = torch.randn(BATCH_SIZE, netG.latent_space_size)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = noise
        fake = netG(noisev)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        # plot various logs
        lib.plot.plot('./results/gan/plots/train critic cost', D_cost.cpu().data.numpy())
        lib.plot.plot('./results/gan/plots/time', time.time() - start_time)
        lib.plot.plot('./results/gan/plots/train gen cost', G_cost.cpu().data.numpy())
        lib.plot.plot('./results/gan/plots/wasserstein distance', Wasserstein_D.cpu().data.numpy())

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            with torch.no_grad():
                dev_disc_costs = []
                for images, _ in iter(dev_gen):
                    images = images.reshape(BATCH_SIZE, 3, 32, 32)
                    imgs = torch.stack([preprocess(item) for item in images])

                    if use_cuda:
                        imgs = imgs.cuda(gpu)
                    imgs_v = imgs

                    D = netD(imgs_v)
                    _dev_disc_cost = -D.mean().cpu().data.numpy()
                    dev_disc_costs.append(_dev_disc_cost)
                print("mean dev_cost: ", np.mean(dev_disc_costs))
                lib.plot.plot('./results/gan/plots/dev critic cost', np.mean(dev_disc_costs))
                generate_image(iteration, netG)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()
        lib.plot.tick()

        # Save GAN model every 1K iterations
        if iteration % 1000 == 999:
            torch.save(gan_model.state_dict(), 'results/gan/model/gan_model.pt')







