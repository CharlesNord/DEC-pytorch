import torch


class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 10),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 784),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z




