from utils.consts import T
import torch
import torch.nn as nn

def denoise(model: nn.Module, x_t: torch.Tensor, alpha: torch.Tensor, end: int= T):
    bs = x_t.shape[0]
    x = x_t.clone()
    for t in reversed(range(end+1)): #do we need to include the 0th example???
        pred_noise = model(x, torch.tensor(t).repeat(bs))
        sqrt_alpha_t = torch.sqrt(alpha[t])
        sqrt_1_minus_alpha_t = torch.sqrt(1 - alpha[t])
        x = (x - sqrt_1_minus_alpha_t * pred_noise) / sqrt_alpha_t
    return x


