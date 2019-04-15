
from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
from torch.functional import F
from torch.optim import Adam
import argparse
from torchvision.utils import save_image
from torch.autograd import Variable



def get_data_loader(dataset_location, batch_size):
    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        utils.download_url(URL + filename, dataset_location)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        # pytorch data loader
        dataset = data_utils.TensorDataset(torch.from_numpy(x))
        dataset_loader = data_utils.DataLoader(x, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc3 = nn.Conv2d(32, 64,kernel_size=3)  
        self.fc4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc5 = nn.Conv2d(64, 256,kernel_size=5) 
        self.fc61 = nn.Linear(256, 100)
        self.fc62 = nn.Linear(256, 100)
        self.fc7 = nn.Linear(100, 256)
        self.fc8 = nn.Conv2d(256, 64, kernel_size=5, padding=4)
        self.fc9 = nn.UpsamplingBilinear2d(scale_factor=2)  
        self.fc10 = nn.Conv2d(64, 32,kernel_size=3, padding=2)  
        self.fc11 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.fc12 = nn.Conv2d(32, 16,kernel_size=3,padding=2)  
        self.fc13 = nn.Conv2d(16, 1, kernel_size=3, padding=2)
        

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        h3 = F.relu(self.fc3(h2))
        h4 = self.fc4(h3)
        h5 = F.relu(self.fc5(h4))
        h5 = h5.view(h5.size(0), -1)
        return self.fc61(h5),  self.fc62(h5)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h6 = F.relu(self.fc7(z))
        h6 = h6.view(h6.size(0), h6.size(1), 1, 1)
        h7 = F.relu(self.fc8(h6))
        h8 = self.fc9(h7)
        h9 = F.relu(self.fc10(h8))
        h10 = self.fc11(h9)
        h11 = F.relu(self.fc12(h10))
        return self.fc13(h11)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return  torch.sigmoid(self.decode(z)), mu, logvar


model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=3e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x,  x,  size_average=False)/x.size(0)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE - KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_.dataset),
                100. * batch_idx / len(train_),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data) in enumerate(valid_):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0),8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(valid_.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    train_, valid_, test_ = get_data_loader("binarized_mnist", args.batch_size)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(128, 100).to(device)
            sample = model.decode(sample)
            save_image(sample.view(128, 1, 28, 28),
                    'results/sample_' + str(epoch) + '.png')
