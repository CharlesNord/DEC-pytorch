import torchvision
import torch
import torch.utils.data as Data
from linear_vae import VAE
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time

weights = torch.load('pretrain_vae.pkl')
vae = VAE().cuda()
vae.load_state_dict(weights)

train_data = torch.load('../mnist/processed/training.pt')
target = train_data[1].numpy()
train_data_init = Variable(train_data[0].unsqueeze(1).type(torch.FloatTensor) / 255.0).cuda()
feat_init, _ = vae.encode(train_data_init.view(-1, 784))

kmeans = KMeans(n_clusters=10, n_init=20)
y_pred_init = kmeans.fit_predict(feat_init.data.cpu().numpy())
cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)).cuda(),
                           requires_grad=True)


def acc(y_pred, y_target):
    D = max(y_pred.max(), y_target.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_target[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum(w[i, j] for i, j in ind) * 1.0 / y_pred.size


print 'Pre-trained auto-encoder accuracy: {}'.format(acc(y_pred_init, target))

alpha = 1.0


def loss_func(feat):
    q = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()

    log_q = torch.log(q)
    loss = F.kl_div(log_q, p)
    return loss, p


loss, p = loss_func(feat_init)

train_data = torchvision.datasets.MNIST('../mnist', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=2)
optimizer = torch.optim.SGD(list(vae.encoder.parameters()) + list(vae.fc1.parameters()) + [cluster_centers], lr=2.0)


def dist_2_label(q_t):
    _, label = torch.max(q_t, dim=1)
    return label.data.cpu().numpy()


for epoch in range(20):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = Variable(batch_x.view(-1, 784)).cuda()
        batch_feat, _ = vae.encode(batch_x)
        batch_loss, _ = loss_func(batch_feat)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print "Epoch: {}\t step: {}\t batch_loss: {}".format(epoch, step, batch_loss.data.cpu()[0])

    feat, _ = vae.encode(train_data_init.view(-1, 784))
    loss, p = loss_func(feat)
    pred_label = dist_2_label(p)
    accuracy = acc(pred_label, target)
    print '========> Epoch: {}\t Accuracy: {}'.format(epoch, accuracy)
    time.sleep(1)
