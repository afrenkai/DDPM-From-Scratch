import torch
import torch.nn as nn
from utils.consts import T
from tqdm import trange, tqdm
def train(model: nn.Module, fake_data: torch.Tensor, alpha: torch.Tensor, end: int, n_epochs: int, bs: int, lr, optimizer: torch.optim.Optimizer, criterion, device):
    for epoch in trange(n_epochs):
        model.train()
        for i in range(0, len(fake_data), bs):
            batch = fake_data[i:i+bs]
            t = torch.randint(1, end+1, (bs,))
            x_0 = batch
            # print(x_0.shape)
            sqrt_alpha_t = torch.sqrt(alpha[t]).unsqueeze(-1)
            sqrt_1_minus_alpha_t = torch.sqrt(1 - alpha[t]).unsqueeze(-1)
            epsilon = torch.randn_like(x_0)
            x_t = sqrt_alpha_t * x_0 + sqrt_1_minus_alpha_t * epsilon

            pred_noise = model(x_t, t)
            loss = criterion(pred_noise, epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} / {n_epochs}, Loss: {loss.item():.4f}")
    
