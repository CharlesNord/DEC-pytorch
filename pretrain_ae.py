import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data as Data
import torchvision
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn.decomposition import PCA

from linear_ae import AE

train_data = torchvision.datasets.MNIST(root='../mnist', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=False)
test_data = torchvision.datasets.MNIST(root='../mnist', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=4)

model = AE().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = torch.nn.BCELoss()


def train(epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (batch_x, _) in enumerate(train_loader):
        data = Variable(batch_x).cuda()
        recon_x, feat = model(data)
        loss = loss_func(recon_x, data.view(-1, 784))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data[0]
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)
            ))

    epoch_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, epoch_loss))
    return epoch_loss


def test(epoch):
    model.eval()
    test_loss = 0
    feat_total = []
    target_total = []
    for i, (data, target) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)  # volatile=True: require_grad=False

        recon_batch, feat = model(data)
        test_loss += loss_func(recon_batch, data.view(-1, 784)).data[0]
        feat_total.append(feat.data.cpu().view(-1, feat.data.shape[1]))
        target_total.append(target)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(-1, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       './pretrain_result/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    feat_total = torch.cat(feat_total, dim=0)
    target_total = torch.cat(target_total, dim=0)
    scatter(feat_total.numpy(), target_total.numpy(), epoch)


def scatter(feat, label, epoch):
    if feat.shape[1] > 2:
        if feat.shape[0] > 5000:
            feat = feat[:5000, :]
            label = label[:5000]
        pca = PCA(n_components=2).fit(feat)
        #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
        #feat = tsne.fit_transform(feat)
        feat = pca.transform(feat)

    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', 10))
    ax = plt.subplot(aspect='equal')
    # sc = ax.scatter(feat[:, 0], feat[:, 1], lw=0, s=40, c=palette[label.astype(np.int)])
    for i in range(10):
        plt.plot(feat[label == i, 0], feat[label == i, 1], '.', c=palette[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.axis('tight')
    for i in range(10):
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

    plt.draw()
    plt.savefig('./pretrain_result/scatter_{}.png'.format(epoch))
    plt.pause(0.001)


if __name__ == '__main__':
    for epoch in range(1, 21):
        train(epoch)
        test(epoch)

    torch.save(model.state_dict(), 'pretrain_ae.pkl')
