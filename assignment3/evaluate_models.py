import argparse
import torch
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from VAE_q3 import ConvEncoder, ConvDecoder, VAE, loss_function


parser = argparse.ArgumentParser(description='VAE for SVHN dataset')
parser.add_argument("--model_path", type=str,
                    help="The path for the model checkpoint")
parser.add_argument('--eps', type=float, default=5)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VAE model (TODO: change hardcoded shapes)
    vae_model = VAE(encoder=ConvEncoder(input_shape=(3, 32, 32)),
                decoder=ConvDecoder(output_shape=(3, 32, 32)))
    vae_model.load_state_dict(torch.load(args.model_path))
    vae_model = vae_model.to(device)


    # Some interesting dimensions visually
    deltas = [-0.75 -0.5, -0.25, 0, 0.25, 0.5, 0.75]
    #dimensions = [5, 10, 31, 36, 70, 76, 77, 81, 82]
    dimensions = range(100)

    # default is zero mean and variance 1
    z_original = torch.randn(1, vae_model.decoder.latent_space_size)
#    save_imaige(z_original, 'eval_result/original.png')

    with torch.no_grad():
        for dim in dimensions:
            sample_dim = torch.Tensor([])
            for delta in deltas:
                z = z_original.clone()
                z[:, dim] = z_original[:, dim] + delta * args.eps
                z = z.to(device)
                sample = vae_model.decode(z).cpu()
                sample_dim = torch.cat((sample_dim, sample))
            save_image(sample_dim, 'eval_result/perturbation_{}.png'.format(dim))
