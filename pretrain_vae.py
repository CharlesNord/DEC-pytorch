import torch
from linear_vae import VAE
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr_scheduler

train_data = torchvision.datasets.MNIST(root='../mnist', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=False)
test_data = torchvision.datasets.MNIST(root='../mnist', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=2)

vae = VAE().cuda()


def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= 784 * 128
    return BCE + KLD


optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, label_train) in enumerate(train_loader):
        data = Variable(data).cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_func(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)
            ))

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss
    ))
    return avg_loss


def test(epoch):
    vae.eval()
    test_loss = 0
    for i, (data, lb) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)  # volatile=True: require_grad=False
        recon_batch, mu, logvar = vae(data)
        test_loss += loss_func(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(128, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       './pretrain_vae/reconstruction_' + str(epoch) + '.png', nrow=n)
    test_loss /= len(test_loader.dataset)
    print('=================================================> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


for epoch in range(20):
    train(epoch)
    val_loss = test(epoch)
    scheduler.step(val_loss)
    torch.save(vae.state_dict(), 'pretrain_vae.pkl')
