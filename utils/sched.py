import torch
import numpy as np

def cos_sched(T, device):
    t = torch.tensor(np.linspace(0, 1, T+1), device = device)
    alpha = torch.cos(t * (np.pi/2))
    alpha = alpha/alpha[0] #starting at 1 like in paper
    return alpha.to(device)
