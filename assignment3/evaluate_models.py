import argparse

import numpy as np
import torch
from torchvision.utils import save_image

from VAE_q3 import ConvEncoder, ConvDecoder, VAE, loss_function
from GAN_q3 import Discriminator, Generator, GAN


def apply_perturbation(model, eps, model_name, z_original):
    with torch.no_grad():
        for dim in range(100):
            result = torch.Tensor([])
            for delta in np.arange(-1, 1.25, 0.25):
                z = z_original.clone()
                z[:, dim] = z_original[:, dim] + delta * eps
                z = z.to(device)
                x = model.decode(z).cpu()
                result = torch.cat((result, x))
            result = result.mul(0.5).add(0.5) # rescale for tanh
            save_image(result, 'eval_result/perturbation/perturbation_{}_{}_{}.png'.format(model_name, dim, eps), nrow=9)

def apply_interpolation(model, model_name, z0, z1):
    with torch.no_grad():
        x_result = torch.Tensor([])
        z_result = torch.Tensor([])
        for alpha in np.arange(0, 1.1, 0.1):
            zalpha = alpha * z0 + (1 - alpha) * z1
            x = model.decode(zalpha.to(device)).cpu()
            x0 = model.decode(z0.to(device)).cpu()
            x1 = model.decode(z1.to(device)).cpu()
            xalpha = alpha * x0 + (1 - alpha) * x1
            z_result = torch.cat((z_result, x))
            x_result = torch.cat((x_result, xalpha))
        z_result = z_result.mul(0.5).add(0.5)  # rescale for tanh
        x_result = x_result.mul(0.5).add(0.5)  # rescale for tanh
        save_image(z_result, 'eval_result/interpolation/interpolation_z_{}.png'.format(model_name), nrow=11)
        save_image(x_result, 'eval_result/interpolation/interpolation_x_{}.png'.format(model_name), nrow=11)


parser = argparse.ArgumentParser(description='VAE and GAN models evaluation')
parser.add_argument("--vae_model_path", type=str,
                    help="The path for the VAE model checkpoint")
parser.add_argument("--gan_model_path", type=str,
                    help="The path for the GAN model checkpoint")
parser.add_argument('--eps', type=float, default=5)
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

    # default distribution is normal with zero mean and variance 1
    z0 = torch.randn(1, VAE_model.decoder.latent_space_size)
    z1 = torch.randn(1, VAE_model.decoder.latent_space_size)

    apply_perturbation(model=VAE_model, eps=args.eps, model_name='VAE', z_original=z0)
    apply_perturbation(model=GAN_model, eps=args.eps, model_name='GAN', z_original=z0)

    apply_interpolation(model=VAE_model, model_name='VAE', z0=z0, z1=z1)
    apply_interpolation(model=GAN_model, model_name='GAN', z0=z0, z1=z1)
