import torch
import torch.nn as nn
import torch.functional as F
from utils.consts import T
from utils.sched import cos_sched
import numpy as np



def ForwardProcess(x_0: torch.Tensor, alpha: torch.Tensor, end: int = T):
    bs = x_0.shape[0]
    print(x_0.shape)
    #assumes x_0 to be of shape bs, sequence length. since original was images, it would have been batch size, channels, height width
    x = x_0.clone()
    x_t = [x]
    for t in range(1, end+1):
        epsilon = torch.randn_like(x)
        sqrt_alpha_t = torch.sqrt(alpha[t])
        sqrt_1_minus_alpha_t = torch.sqrt(1-alpha[t])
        x = sqrt_alpha_t * x + sqrt_1_minus_alpha_t * epsilon
        x_t.append(x)
    return torch.stack(x_t, dim = 1) #bs, T+1, inp_dim)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = cos_sched(T, device)
    print(device)
    x_0 = torch.rand(100, 6).to(device)

    print(ForwardProcess(x_0, alpha))
