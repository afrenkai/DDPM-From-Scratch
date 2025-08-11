import torch.nn as nn
import torch
class UNetDenoise(nn.Module):
    def __init__(self, inp_dim: int, time_dim: int, hidden_dim: int, steps: int, device):
        super().__init__() 
        self.device = device
        self.time_emb = nn.Embedding(steps, time_dim, device = device)
        self.time_proj = nn.Linear(time_dim, hidden_dim, device = device)
        self.reshape_proj_1 = nn.Linear(inp_dim + hidden_dim, inp_dim + hidden_dim, device = device) #timexinp -> time x inp+hid
        self.res_1_enc = nn.Linear(inp_dim + hidden_dim , hidden_dim, device = device) #time x inp+hid -> time x hid
        self.res_1_relu = nn.ReLU()
        self.res_1_resid = nn.Linear(hidden_dim, hidden_dim, device = device)



        self.res_2_enc = nn.Linear(hidden_dim, hidden_dim, device = device)
        self.res_2_relu = nn.ReLU()
        self.res_2_resid = nn.Linear(hidden_dim, hidden_dim, device = device)
        self.res_3_enc = nn.Linear(hidden_dim, hidden_dim, device = device)
        self.res_3_relu = nn.ReLU()
        self.res_3_resid = nn.Linear(hidden_dim, inp_dim, device = device)
    
    def forward(self, x_t, t):
        x_t = x_t.to(self.device)
        x_t = x_t.detach().clone().float()
        t = t.to(self.device) 
        t_emb = self.time_emb(t)
        t_proj = self.time_proj(t_emb)
        x_t_t = torch.cat([x_t, t_proj], dim = 1)
        # print(x_t_t.shape)
        x = self.reshape_proj_1(x_t_t) 
        # print(x.shape)
        x = self.res_1_relu(self.res_1_enc(x))
        # print(x.shape)
        x = self.res_1_resid(x)
        # print(x.shape)

        # print("res_1")
        x = self.res_2_resid(self.res_2_relu(self.res_2_enc(x)))
        x = self.res_3_resid(self.res_3_relu(self.res_3_enc(x)))
        return x
