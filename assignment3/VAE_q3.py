# Inspired from https://github.com/pytorch/examples/blob/master/vae/main.py


import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, latent_space_size=100):
        super(ConvEncoder, self).__init__()

        self.input_shape = input_shape
        self.latent_space_size = latent_space_size

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64, 256, kernel_size=5)
        self.fc61 = nn.Linear(256, latent_space_size)
        self.fc62 = nn.Linear(256, latent_space_size)

    def forward(self, x):
        h1 = F.elu(self.conv1(x))
        h2 = self.pool2(h1)
        h3 = F.elu(self.conv3(h2))
        h4 = self.pool4(h3)
        h5 = F.elu(self.conv5(h4))
        h6 = h5.view(h5.size(0), -1)
        return self.fc61(h6), self.fc62(h6)

class ConvDecoder(nn.Module):
    def __init__(self, output_shape, latent_space_size=100):
        super(ConvDecoder, self).__init__()

        self.latent_space_size = latent_space_size
        self.output_shape = output_shape

        self.fc7 = nn.Linear(latent_space_size, 256)
        self.conv8 = nn.Conv2d(256, 64, kernel_size=5, padding=4)
        self.up9 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv10 = nn.Conv2d(64, 32, kernel_size=3, padding=3)
        self.conv11 = nn.Conv2d(32, 16, kernel_size=3, padding=3)
        self.up12 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv13 = nn.Conv2d(16, 16, kernel_size=3, padding=0)
        self.conv14 = nn.Conv2d(16, output_shape[0], kernel_size=3, padding=0)

    def forward(self, z):
        h6 = F.leaky_relu(self.fc7(z))
        h6 = h6.view(h6.size(0), h6.size(1), 1, 1)
        h7 = F.leaky_relu(self.conv8(h6))
        h8 = self.up9(h7)
        h9 = F.leaky_relu(self.conv10(h8))
        h10 = F.leaky_relu(self.conv11(h9))
        h11 = self.up12(h10)
        h12 = F.leaky_relu(self.conv13(h11))
        out = F.tanh(self.conv14(h12))
        return out


class MLPEncoder(nn.Module):
    def __init__(self, input_shape, latent_space_size=100, hidden_layer_size=400):
        super(MLPEncoder, self).__init__()

        self.input_shape = input_shape
        self.flat_size = input_shape[1] * input_shape[2]
        self.latent_space_size = latent_space_size

        self.fc1 = nn.Linear(self.flat_size, hidden_layer_size)
        self.fc21 = nn.Linear(hidden_layer_size, latent_space_size)
        self.fc22 = nn.Linear(hidden_layer_size, latent_space_size)

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.flat_size)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

class MLPDecoder(nn.Module):
    def __init__(self, output_shape, latent_space_size=100, hidden_layer_size=400):
        super(MLPDecoder, self).__init__()

        self.output_shape = output_shape
        self.flat_size = output_shape[1] * output_shape[2]
        self.latent_space_size = latent_space_size

        self.fc3 = nn.Linear(latent_space_size, hidden_layer_size)
        self.fc4 = nn.Linear(hidden_layer_size, self.flat_size)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(h3))
        return out.view(-1, self.output_shape[0], self.output_shape[1], self.output_shape[2])


class VAE(nn.Module):
    def __init__(self, encoder, decoder, n_channels=3):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert encoder.latent_space_size == decoder.latent_space_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        recon_batch = recon_batch.mul(0.5).add(0.5)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    train_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            recon_batch = recon_batch.mul(0.5).add(0.5)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch[:n]])
                save_image(comparison.cpu(),
                           'results/vae/reconstruction/reconstruction_{}.png'.format(epoch), nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


parser = argparse.ArgumentParser(description='VAE for SVHN dataset')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data', split='train', download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data', split='test', transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # VAE model (TODO: change hardcoded shapes)
    model = VAE(encoder=ConvEncoder(input_shape=(3, 32, 32)),
                decoder=ConvDecoder(output_shape=(3, 32, 32)))
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        with torch.no_grad():
            z = torch.randn(128, model.decoder.latent_space_size)
            z = z.to(device)
            sample = model.decode(z).cpu()
            sample = sample.mul(0.5).add(0.5)
            save_image(sample, 'results/vae/samples/sample_' + str(epoch) + '.png')

        if test_loss < best_loss:
            # save model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), 'results/vae/model/vae_model.pt')
            best_loss = test_loss
            patience = 0
        elif patience <= 3:
            patience += 1
            print('Patience {}'.format(patience))
        else:
            print('Early stopping after {} epochs'.format(epoch))
            break
