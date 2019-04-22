import argparse
from collections import OrderedDict

import numpy as np
import torch
from torchvision.utils import save_image
import tqdm
from VAE_q3 import ConvEncoder, ConvDecoder, VAE, loss_function
from GAN_q3 import Discriminator, Generator, GAN



parser = argparse.ArgumentParser(description='Generate samples from VAE and GAN')
parser.add_argument("--vae_model_path", type=str,
                    help="The path for the VAE model checkpoint")
parser.add_argument("--gan_model_path", type=str,
                    help="The path for the GAN model checkpoint")
parser.add_argument('--n_samples', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VAE model
    VAE_model = VAE(encoder=ConvEncoder(input_shape=(3, 32, 32)),
                    decoder=ConvDecoder(output_shape=(3, 32, 32)))
    VAE_model.load_state_dict(torch.load(args.vae_model_path, map_location = lambda storage, loc: storage))
    VAE_model = VAE_model.to(device)

    # WGAN-GP model
    GAN_model = GAN(critic=Discriminator(input_shape=(3, 32, 32)),
                    generator=Generator(output_shape=(3, 32, 32)))

    GAN_model.load_state_dict(torch.load(args.gan_model_path, map_location=lambda storage, loc: storage))
    GAN_model = GAN_model.to(device)

    for i in tqdm.tqdm(range(args.n_samples)):
        # default distribution is normal with zero mean and variance 1
        z = torch.randn(1, VAE_model.decoder.latent_space_size)
        x_vae = VAE_model.decode(z.to(device)).cpu()
        x_gan = GAN_model.generator(z.to(device)).cpu()
        x_vae = x_vae.mul(0.5).add(0.5) # rescale for tanh
        x_gan = x_gan.mul(0.5).add(0.5) # rescale for tanh
        save_image(x_vae, 'results/vae/vae_single_samples/{}.png'.format(i))
        save_image(x_gan, 'results/gan/gan_single_samples/{}.png'.format(i))