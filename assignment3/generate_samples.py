import argparse

import numpy as np
import torch
from torchvision.utils import save_image

from VAE_q3 import ConvEncoder, ConvDecoder, VAE, loss_function


parser = argparse.ArgumentParser(description='Generate samples from VAE')
parser.add_argument("--vae_model_path", type=str,
                    help="The path for the VAE model checkpoint")
parser.add_argument('--n_samples', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VAE model
    VAE_model = VAE(encoder=ConvEncoder(input_shape=(3, 32, 32)),
                    decoder=ConvDecoder(output_shape=(3, 32, 32)))
    VAE_model.load_state_dict(torch.load(args.vae_model_path))
    VAE_model = VAE_model.to(device)
    model_name = 'VAE'

    for i in range(args.n_samples):
        # default distribution is normal with zero mean and variance 1
        z = torch.randn(1, VAE_model.decoder.latent_space_size)
        x = VAE_model.decode(z.to(device)).cpu()
        save_image(x, '{}_samples/samples/{}.png'.format(model_name, i))
