from utils.datautils import *
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, 500)
        self.fc21 = nn.Linear(500, latent_size)
        self.fc22 = nn.Linear(500, latent_size)
        self.fc3 = nn.Linear(latent_size, 500)
        self.fc4 = nn.Linear(500, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    print('Logvar:')
    print(logvar)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print('KLD:')
    print(KLD)

    return BCE + KLD


def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('Loss: {}'.format(loss.item()))
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


(data, labels), (input_size, num_classes) = load_tcga_data(ImputationType.DROP_SAMPLES)
folds, labelled_indices, val_test_split = pickle.load(open('./data/tcga/10000_labelled_5_folds_drop_samples.p'))

for i, (train_indices, test_val_indices) in enumerate(folds):
    normalizer = GaussianNormalizeTensors()

    train_data = normalizer.apply_train(data[train_indices])
    u_d = TensorDataset(train_data, -1 * torch.ones(train_data.size(0)))
    u_dl = DataLoader(u_d, batch_size=100, shuffle=True)
    test_val_data = normalizer.apply_test(data[test_val_indices])

    model = VAE(input_size, 100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        train(epoch, model, optimizer, u_dl)
