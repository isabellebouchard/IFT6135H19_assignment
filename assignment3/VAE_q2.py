# Inspired from https://github.com/pytorch/examples/blob/master/vae/main.py , 
# from http://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/, and
# from https://github.com/bjlkeng/sandbox/tree/master/notebooks/vae-importance_sampling


from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.functional import F
from torch.optim import Adam
import argparse
from torchvision.utils import save_image
from scipy.stats import norm
from scipy.special import logsumexp

loss_=[]

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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64,kernel_size=3)  
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 256,kernel_size=5) 
        self.fc11 = nn.Linear(256, 100)
        self.fc12 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 256)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=5, padding=4)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)  
        self.conv5 = nn.Conv2d(64, 32,kernel_size=3, padding=2)  
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv6 = nn.Conv2d(32, 16,kernel_size=3,padding=2)  
        self.conv7 = nn.Conv2d(16, 1, kernel_size=3, padding=2)
        

    def encode(self, x):
        h1 = F.elu(self.conv1(x))
        h2 = self.pool1(h1)
        h3 = F.elu(self.conv2(h2))
        h4 = self.pool2(h3)
        h5 = F.elu(self.conv3(h4))
        h5 = h5.view(h5.size(0), -1)
        return self.fc11(h5),  self.fc12(h5)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h6 = F.elu(self.fc2(z))
        h6 = h6.view(h6.size(0), h6.size(1), 1, 1)
        h7 = F.elu(self.conv4(h6))
        h8 = self.up1(h7)
        h9 = F.elu(self.conv5(h8))
        h10 = self.up2(h9)
        h11 = F.elu(self.conv6(h10))
        return self.conv7(h11)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return  torch.sigmoid(self.decode(z)), mu, logvar



def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD =  -0.5 * torch.sum(1 + logvar - mu.pow(2)-logvar.exp())
    return BCE + KLD


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
    loss_.append(train_loss / len(train_.dataset))
    return np.array(loss_)


def valid(epoch):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, (data) in enumerate(valid_):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            valid_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0),8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    valid_loss /= len(valid_.dataset)
    print('====> Validation set loss: {:.4f}'.format(valid_loss))
    
    
    
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data) in enumerate(test_):
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
    
    
    
def importance_sampling( data, num_samples):
    #mean and variance from the encoder
    mu, logvar = model.encode(data)
    #some initializations
    z_samples = []
    qz_x = []
    result = []
    mu = mu.to(torch.device("cpu")).detach().numpy()
    logvar = logvar.to(torch.device("cpu")).detach().numpy()
    
    #Sampling z_ki and calculating qz_x and pz
    for m, s in zip(mu, logvar):
        z_vals = [np.random.normal(m[i], np.exp(s[i]), num_samples)
                  for i in range(len(m))]
        qz_x_vals = [norm.pdf(z_vals[i], loc=m[i], scale=np.exp(s[i]))
                  for i in range(len(m))]
        z_samples.append(z_vals)
        qz_x.append(qz_x_vals)
    
    z_samples = np.array(z_samples)
    pz = norm.pdf(z_samples)
    qz_x = np.array(qz_x)
    
    z_samples = np.swapaxes(z_samples, 1, 2)
    pz = np.swapaxes(pz, 1, 2)
    qz_x = np.swapaxes(qz_x, 1, 2)
    
    z_samples = torch.from_numpy(z_samples).to(device).float()
    
    # For each input i in the batch_size data calculates logpx_z, logpz and logqz_x
    # Also estimate the logpx_i
    for i in range(len(data)):
        datsin = data[i].reshape(784)
        datsin = datsin.to(torch.device("cpu")).numpy()
        x_predict =  torch.sigmoid(model.decode(z_samples[i]))
        x_predict = x_predict.to(torch.device("cpu")).detach().numpy().reshape(-1,784)
        logpx_z = np.sum(datsin * np.log(x_predict) + (1. - datsin) * np.log(1.0 - x_predict), axis=-1)
        logpz = np.sum(np.log(pz[i]), axis=-1)
        logqz_x = np.sum(np.log(qz_x[i]), axis=-1)
        argsum = logpx_z*logpz/logqz_x
        logpx = -np.log(num_samples) + logsumexp(argsum)
        result.append(logpx)
    
    #Estimates the logpx for the batch size    
    average = np.mean(result)
    return np.array(result), average

    
    
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

model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=3e-4)       

if __name__ == "__main__":
    logtotal_val=[]
    logtotal_tes=[]
    train_model = False
    train_, valid_, test_ = get_data_loader("binarized_mnist", args.batch_size)
    if train_model==True:
       
        model = model.to(device)
        for epoch in range(1, args.epochs + 1):
            loss = train(epoch)
            with torch.no_grad():
                sample = torch.randn(128, 100).to(device)
                sample = model.decode(sample)
                save_image(sample.view(128, 1, 28, 28),
                               'results/sample_' + str(epoch) + '.png')      
        torch.save(model.state_dict(), 'results/model.pt')
        print(loss)
    if train_model ==False:
        model.load_state_dict(torch.load('results/model.pt'))
        model=model.to(device)
        args.epochs = 1
        for epoch in range(1, args.epochs + 1):
            valid(epoch)
            test(epoch)
            for i, (data) in enumerate(valid_):
                    data=data.to(device)
                    logpx,logpx_batch = importance_sampling(data, 200)
                    logtotal_val = np.concatenate((logtotal_val,logpx_batch), axis=None)
            print(np.mean(logtotal_val))
            for i, (data) in enumerate(test_):
                    data=data.to(device)
                    logpx,logpx_batch = importance_sampling(data, 200)
                    logtotal_tes=np.concatenate((logtotal_tes,logpx_batch), axis=None)
            print(np.mean(logtotal_tes))