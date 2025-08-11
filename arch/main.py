import torch
import torch.nn as nn
from utils.sched import cos_sched
from utils.consts import T
from arch.mod import UNetDenoise
from arch.fwd import ForwardProcess
from arch.denoise import denoise
from arch.train import train
import matplotlib.pyplot as plt
import torch.distributions as D


def main(plot: bool = False, kl: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    lr = 1e-3
    alpha = cos_sched(T, device)
    torch.manual_seed(67)
    fake_data = torch.rand(10000,6).to(device)
    inp_dim = fake_data.shape[1]
    # print(inp_dim)
    model = UNetDenoise(fake_data.shape[1], 16, 64, T, device = device)
    train(model, fake_data, alpha, T, 10, 16, lr, torch.optim.AdamW(model.parameters(), lr = lr), nn.MSELoss(), device)
    x_t = torch.randn(10000, 6).to(device)
    x_denoised = denoise(model, x_t, alpha, T)
    if plot:
        plt.figure(figsize=(10, 4))
        plt.hist(fake_data.cpu().numpy(), bins=20, alpha=0.5, label="Original Data")
        plt.hist(x_denoised.cpu().numpy(), bins=20, alpha=0.5, label="Denoised Data")
        plt.legend()
        plt.title("Generated Data vs Original Data")
        plt.show()
    if kl:
        mu_0 = torch.mean(fake_data, dim = 0)
        sigma_0 = torch.std(fake_data, dim = 0, unbiased = False)
        mu_denoised = torch.mean(x_denoised, dim = 0)
        sigma_denoised = torch.mean(x_denoised, dim = 0, unbiased = False)
        p = D.normal(loc = mu_0, scale = sigma_0)
        q = D.normal(loc = mu_denoised, scale = sigma_denoised)
        kl = D.kl_divergence(p, q).item()
        print(kl)
    print(nn.MSE(fake_data, x_denoised))

        



if __name__ == "__main__":
    main(plot=True)
